"""Native PyTorch embedding sharding — fallback for FBGEMM TBE ROCm issues.

Provides ShardedEmbeddingCollection: shards embedding tables across GPUs using
standard nn.Embedding + torch.distributed.all_to_all_single. Same forward API
as TorchRecEmbeddings (dict[str, Tensor] -> dict[str, Tensor]).

Supports table-wise and row-wise sharding. Gradients flow through a custom
autograd Function wrapping all-to-all so embedding optimizer runs locally on
each GPU's shard.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from primus_dlrm.config import EmbeddingTableConfig

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    strategy: Literal["table_wise", "row_wise"]
    owner_rank: int = 0
    num_shards: int = 1


class _AllToAllEmb(torch.autograd.Function):
    """Differentiable all-to-all for embedding vectors (backward reverses splits).

    Forward:  out = all_to_all(inp), where inp is split by in_splits and
              out is formed by concatenating out_splits-sized chunks from each rank.
    Backward: grad_inp = all_to_all(grad_out), with splits swapped.
    """

    @staticmethod
    def forward(ctx, inp, out_splits, in_splits, pg):
        ctx.pg = pg
        ctx.fwd_in_splits = list(in_splits)
        ctx.fwd_out_splits = list(out_splits)
        out = torch.empty(
            sum(out_splits), *inp.shape[1:],
            dtype=inp.dtype, device=inp.device,
        )
        dist.all_to_all_single(out, inp.contiguous(), out_splits, in_splits, group=pg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        bwd_out_splits = ctx.fwd_in_splits
        bwd_in_splits = ctx.fwd_out_splits
        grad_in = torch.empty(
            sum(bwd_out_splits), *grad_out.shape[1:],
            dtype=grad_out.dtype, device=grad_out.device,
        )
        dist.all_to_all_single(
            grad_in, grad_out, bwd_out_splits, bwd_in_splits, group=ctx.pg,
        )
        return grad_in, None, None, None


def _all_to_all_emb(inp: Tensor, out_splits: list[int], in_splits: list[int],
                    pg: dist.ProcessGroup) -> Tensor:
    """Differentiable all-to-all for float embedding vectors."""
    return _AllToAllEmb.apply(inp, out_splits, in_splits, pg)


def _all_to_all_ids(inp: Tensor, out_splits: list[int], in_splits: list[int],
                    pg: dist.ProcessGroup) -> Tensor:
    """Non-differentiable all-to-all for integer IDs (no gradient needed)."""
    out = torch.empty(sum(out_splits), dtype=inp.dtype, device=inp.device)
    dist.all_to_all_single(out, inp.contiguous(), out_splits, in_splits, group=pg)
    return out


class ShardedEmbeddingCollection(nn.Module):
    """Embedding tables sharded across GPUs with all-to-all communication.

    Same forward API as TorchRecEmbeddings::

        results = shard_emb({"uid": ids, "item": ids, ...})

    Each GPU only stores nn.Embedding parameters for its local shards.
    All-to-all collectives redistribute lookups during forward/backward.
    """

    def __init__(
        self,
        table_specs: list[EmbeddingTableConfig],
        shard_configs: dict[str, ShardConfig],
        pg: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if pg is None:
            pg = dist.group.WORLD
        self.pg = pg
        self.world_size = dist.get_world_size(pg)
        self.rank = dist.get_rank(pg)

        if device is None:
            device = torch.device(f"cuda:{self.rank}")
        self.device = device

        self._feature_to_table: dict[str, str] = {}
        self._table_pooling: dict[str, str] = {}
        self._feature_dims: dict[str, int] = {}
        self._shard_configs: dict[str, ShardConfig] = shard_configs

        self.local_embeddings = nn.ModuleDict()
        self._table_specs: dict[str, EmbeddingTableConfig] = {}
        self._local_num_rows: dict[str, int] = {}

        for spec in table_specs:
            self._table_specs[spec.name] = spec
            for feat in spec.features:
                self._feature_to_table[feat] = spec.name
                self._feature_dims[feat] = spec.embedding_dim
            self._table_pooling[spec.name] = spec.pooling

            sc = shard_configs.get(spec.name)
            if sc is None:
                sc = ShardConfig(strategy="table_wise", owner_rank=0)
                shard_configs[spec.name] = sc

            if sc.strategy == "table_wise":
                if sc.owner_rank == self.rank:
                    self.local_embeddings[spec.name] = nn.Embedding(
                        spec.num_embeddings, spec.embedding_dim, device=device,
                    )
                    self._local_num_rows[spec.name] = spec.num_embeddings
            elif sc.strategy == "row_wise":
                chunk = math.ceil(spec.num_embeddings / self.world_size)
                start = self.rank * chunk
                local_rows = min(chunk, max(0, spec.num_embeddings - start))
                if local_rows > 0:
                    self.local_embeddings[spec.name] = nn.Embedding(
                        local_rows, spec.embedding_dim, device=device,
                    )
                self._local_num_rows[spec.name] = local_rows

    def forward(
        self,
        features: dict[str, Tensor],
        padding_idx: int = 0,
    ) -> dict[str, Tensor]:
        results: dict[str, Tensor] = {}

        tw_tables: dict[str, list[str]] = {}
        rw_tables: dict[str, list[str]] = {}

        for feat_name, ids in features.items():
            table_name = self._feature_to_table[feat_name]
            sc = self._shard_configs[table_name]
            if sc.strategy == "table_wise":
                tw_tables.setdefault(table_name, []).append(feat_name)
            elif sc.strategy == "row_wise":
                rw_tables.setdefault(table_name, []).append(feat_name)

        for table_name, feat_names in tw_tables.items():
            self._forward_table_wise(table_name, feat_names, features, results, padding_idx)

        for table_name, feat_names in rw_tables.items():
            self._forward_row_wise(table_name, feat_names, features, results, padding_idx)

        return results

    # ---- Table-wise ----

    def _forward_table_wise(
        self, table_name: str, feat_names: list[str],
        features: dict[str, Tensor], results: dict[str, Tensor],
        padding_idx: int,
    ) -> None:
        sc = self._shard_configs[table_name]
        spec = self._table_specs[table_name]
        owner = sc.owner_rank
        D = spec.embedding_dim
        pooling = self._table_pooling[table_name]

        for feat_name in feat_names:
            ids = features[feat_name]
            orig_shape = ids.shape

            flat_ids = ids.reshape(-1).contiguous()
            N = flat_ids.shape[0]

            counts_send = [0] * self.world_size
            counts_send[owner] = N

            counts_recv_tensor = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
            counts_send_tensor = torch.tensor(counts_send, dtype=torch.long, device=self.device)
            dist.all_to_all_single(counts_recv_tensor, counts_send_tensor, group=self.pg)
            counts_recv = [int(x) for x in counts_recv_tensor.tolist()]

            all_ids = _all_to_all_ids(flat_ids.long(), counts_recv, counts_send, self.pg)

            total_recv = sum(counts_recv)
            if self.rank == owner:
                emb_table = self.local_embeddings[table_name]
                all_embs = emb_table(all_ids)
            else:
                all_embs = torch.zeros(
                    total_recv, D, dtype=torch.float32, device=self.device,
                    requires_grad=True,
                )

            send_back_counts = counts_recv
            recv_back_counts = counts_send
            my_embs = _all_to_all_emb(all_embs, recv_back_counts, send_back_counts, self.pg)

            if pooling in ("mean", "sum"):
                emb_view = my_embs.view(*orig_shape, D)
                mask = (ids != padding_idx).unsqueeze(-1).float()
                if pooling == "sum":
                    results[feat_name] = (emb_view * mask).sum(dim=-2)
                else:
                    denom = mask.sum(dim=-2).clamp(min=1)
                    results[feat_name] = (emb_view * mask).sum(dim=-2) / denom
            else:
                if len(orig_shape) == 1:
                    results[feat_name] = my_embs.view(orig_shape[0], D)
                else:
                    results[feat_name] = my_embs.view(*orig_shape, D)

    # ---- Row-wise ----

    def _forward_row_wise(
        self, table_name: str, feat_names: list[str],
        features: dict[str, Tensor], results: dict[str, Tensor],
        padding_idx: int,
    ) -> None:
        spec = self._table_specs[table_name]
        D = spec.embedding_dim
        pooling = self._table_pooling[table_name]
        chunk = math.ceil(spec.num_embeddings / self.world_size)

        for feat_name in feat_names:
            ids = features[feat_name]
            orig_shape = ids.shape
            flat_ids = ids.reshape(-1).contiguous()
            N = flat_ids.shape[0]

            dest_rank = (flat_ids // chunk).clamp(max=self.world_size - 1)
            local_ids = flat_ids - dest_rank * chunk

            sort_idx = dest_rank.argsort()
            sorted_dest = dest_rank[sort_idx]
            sorted_local_ids = local_ids[sort_idx].contiguous()

            send_counts = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
            for r in range(self.world_size):
                send_counts[r] = (sorted_dest == r).sum()
            send_counts_list = [int(x) for x in send_counts.tolist()]

            recv_counts = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
            dist.all_to_all_single(recv_counts, send_counts, group=self.pg)
            recv_counts_list = [int(x) for x in recv_counts.tolist()]

            received_ids = _all_to_all_ids(
                sorted_local_ids.long(), recv_counts_list, send_counts_list, self.pg,
            )

            total_recv = sum(recv_counts_list)
            if table_name in self.local_embeddings and total_recv > 0:
                emb_table = self.local_embeddings[table_name]
                safe_ids = received_ids.clamp(0, self._local_num_rows[table_name] - 1)
                looked_up = emb_table(safe_ids)
            else:
                looked_up = torch.zeros(
                    total_recv, D,
                    dtype=torch.float32, device=self.device,
                    requires_grad=True,
                )

            my_embs_sorted = _all_to_all_emb(
                looked_up, send_counts_list, recv_counts_list, self.pg,
            )

            unsort_idx = sort_idx.argsort()
            my_embs = my_embs_sorted[unsort_idx]

            if pooling in ("mean", "sum"):
                emb_view = my_embs.view(*orig_shape, D)
                mask = (ids != padding_idx).unsqueeze(-1).float()
                if pooling == "sum":
                    results[feat_name] = (emb_view * mask).sum(dim=-2)
                else:
                    denom = mask.sum(dim=-2).clamp(min=1)
                    results[feat_name] = (emb_view * mask).sum(dim=-2) / denom
            else:
                if len(orig_shape) == 1:
                    results[feat_name] = my_embs.view(orig_shape[0], D)
                else:
                    results[feat_name] = my_embs.view(*orig_shape, D)


def assign_sharding(
    table_specs: list[EmbeddingTableConfig],
    world_size: int,
    strategy: str = "auto",
) -> dict[str, ShardConfig]:
    """Assign sharding config to each table.

    ``"auto"``: tables < 10MB → table_wise (round-robin), else → row_wise.
    ``"table_wise"``: force all tables to table_wise.
    ``"row_wise"``: force all tables to row_wise.
    """
    configs: dict[str, ShardConfig] = {}

    if strategy == "table_wise":
        for i, spec in enumerate(table_specs):
            configs[spec.name] = ShardConfig("table_wise", owner_rank=i % world_size)
        return configs

    if strategy == "row_wise":
        for spec in table_specs:
            configs[spec.name] = ShardConfig("row_wise", num_shards=world_size)
        return configs

    tw_idx = 0
    for spec in table_specs:
        table_bytes = spec.num_embeddings * spec.embedding_dim * 4
        if table_bytes < 10 * 1024 * 1024:
            configs[spec.name] = ShardConfig("table_wise", owner_rank=tw_idx % world_size)
            tw_idx += 1
        else:
            configs[spec.name] = ShardConfig("row_wise", num_shards=world_size)

    return configs
