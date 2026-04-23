"""OneTrans: unified Transformer for sequence modeling + feature interaction.

Uses split-history pools (listen+, like, skip) with per-behavior MLP tokenizers.
Supports optional counter features and in-batch contrastive loss.
All categorical embeddings backed by TorchRec EmbeddingCollection.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from primus_dlrm.config import Config
from primus_dlrm.models.base import BaseModel
from primus_dlrm.models.embedding import TorchRecEmbeddings

try:
    from primus_turbo.pytorch.ops import flash_attn_func as _turbo_flash_attn
    _HAS_TURBO_ATTN = True
except ImportError:
    _HAS_TURBO_ATTN = False

try:
    from flash_attn import flash_attn_func as _flash_attn
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_4_raw
    import torch._dynamo
    @torch._dynamo.disable
    def _flash_attn_4(q, k, v, causal=True):
        return _flash_attn_4_raw(q, k, v, causal=causal)[0]
    _HAS_FLASH_ATTN_4 = True
except ImportError:
    _HAS_FLASH_ATTN_4 = False



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@torch.fx.wrap
def pyramid_schedule(l_s: int, l_ns: int, n_layers: int) -> list[int]:
    """Number of S-token queries at each layer, linearly shrinking."""
    if n_layers == 1:
        return [l_ns]
    schedule = []
    for l in range(n_layers):
        q = l_s - l * (l_s - l_ns) / (n_layers - 1)
        schedule.append(max(int(round(q)), l_ns))
    return schedule


def build_pyramid_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    """Causal mask [q_len, kv_len] accounting for query offset."""
    offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1) + offset
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= q_idx


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class MixedAttention(nn.Module):
    """Multi-head attention with shared S-token projections and
    batched token-specific NS-token projections."""

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int,
                 dropout: float = 0.0, attention_impl: str = "sdpa"):
        super().__init__()
        if attention_impl == "turbo" and not _HAS_TURBO_ATTN:
            raise RuntimeError(
                "attention_impl='turbo' requires primus_turbo package. "
            )
        if attention_impl == "fav2" and not _HAS_FLASH_ATTN:
            raise RuntimeError(
                "attention_impl='fav2' requires flash_attn package. "
            )
        if attention_impl == "fav4" and not _HAS_FLASH_ATTN_4:
            raise RuntimeError(
                "attention_impl='fav4' requires flash-attn-4 package "
                "(pip install flash-attn-4). Optimized for Blackwell GPUs."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_ns = n_ns_tokens
        self.attention_impl = attention_impl

        self.s_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.ns_qkv_weight = nn.Parameter(torch.empty(n_ns_tokens, 3 * d_model, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.s_qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for i in range(self.n_ns):
            nn.init.xavier_uniform_(self.ns_qkv_weight.data[i])

    def forward(self, x: Tensor, n_s: int, mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape
        s_part = x[:, :n_s]
        ns_part = x[:, n_s:]

        s_qkv = self.s_qkv(s_part).reshape(B, n_s, 3, self.n_heads, self.d_head)
        s_q, s_k, s_v = s_qkv.unbind(2)

        ns_qkv = torch.einsum("btn,tmn->btm", ns_part, self.ns_qkv_weight)
        ns_qkv = ns_qkv.reshape(B, self.n_ns, 3, self.n_heads, self.d_head)
        ns_q, ns_k, ns_v = ns_qkv.unbind(2)

        # Q/K/V in BSHD: [B, seq, n_heads, d_head]
        q = torch.cat([s_q, ns_q], dim=1)
        k = torch.cat([s_k, ns_k], dim=1)
        v = torch.cat([s_v, ns_v], dim=1)

        if mask is not None:
            q_len = mask.shape[0]
            if q_len < T:
                q = q[:, T - q_len:]

        drop_p = self.attn_drop if self.training else 0.0

        if self.attention_impl == "turbo":
            out = _turbo_flash_attn(q, k, v, causal=True, dropout_p=drop_p)
        elif self.attention_impl == "fav4":
            out = _flash_attn_4(q, k, v, causal=True)
        elif self.attention_impl == "fav2":
            out = _flash_attn(q, k, v, causal=True, dropout_p=drop_p)
        elif self.attention_impl == "sdpa":
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_mask = mask.unsqueeze(0).unsqueeze(0) if mask is not None else None
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=drop_p,
            ).transpose(1, 2)
        else:
            raise ValueError(
                f"Unknown attention_impl: {self.attention_impl!r}. "
                f"Supported: 'sdpa', 'fav2', 'fav4', 'turbo'"
            )

        out = out.reshape(B, out.shape[1], self.d_model)
        return self.out_proj(out)


class MixedFFN(nn.Module):
    """FFN with shared weights for S-tokens and batched token-specific weights for NS-tokens."""

    def __init__(self, d_model: int, ffn_dim: int, n_ns_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.n_ns = n_ns_tokens
        self.d_model = d_model

        self.s_w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.s_w2 = nn.Linear(ffn_dim, d_model, bias=False)

        self.ns_w1 = nn.Parameter(torch.empty(n_ns_tokens, ffn_dim, d_model))
        self.ns_w2 = nn.Parameter(torch.empty(n_ns_tokens, d_model, ffn_dim))

        self.dropout = nn.Dropout(dropout)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.s_w1.weight)
        nn.init.xavier_uniform_(self.s_w2.weight)
        for i in range(self.n_ns):
            nn.init.xavier_uniform_(self.ns_w1.data[i])
            nn.init.xavier_uniform_(self.ns_w2.data[i])

    def forward(self, x: Tensor, n_s: int) -> Tensor:
        s_part = x[:, :n_s]
        ns_part = x[:, n_s:]

        s_out = self.s_w2(self.dropout(F.gelu(self.s_w1(s_part))))
        ns_h = F.gelu(torch.einsum("btn,tmn->btm", ns_part, self.ns_w1))
        ns_out = torch.einsum("btm,tnm->btn", self.dropout(ns_h), self.ns_w2)

        return torch.cat([s_out, ns_out], dim=1)


class OneTransBlock(nn.Module):
    """Pre-norm causal Transformer block with mixed parameterization."""

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int,
                 ffn_mult: int = 4, dropout: float = 0.0,
                 attention_impl: str = "sdpa"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MixedAttention(d_model, n_heads, n_ns_tokens, dropout,
                                   attention_impl=attention_impl)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MixedFFN(d_model, d_model * ffn_mult, n_ns_tokens, dropout)

    def forward(self, x: Tensor, mask: Tensor, q_len: int, n_s_tokens: int) -> Tensor:
        kv_len = x.shape[1]
        n_ns = kv_len - n_s_tokens

        normed = self.norm1(x)
        attn_out = self.attn(normed, n_s=n_s_tokens, mask=mask)
        x_tail = x[:, kv_len - q_len:]
        z = attn_out + x_tail

        n_s_out = q_len - n_ns
        ffn_out = self.ffn(self.norm2(z), n_s=n_s_out)
        return z + ffn_out


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class SequentialTokenizer(nn.Module):
    """Projects per-position history embeddings to S-tokens."""

    def __init__(self, raw_dim: int, d_model: int, max_len: int = 1024,
                 use_pos_embed: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(max_len, d_model) if use_pos_embed else None

    def forward(self, raw: Tensor) -> Tensor:
        """raw: [B, L, raw_dim] -> [B, L, d_model]"""
        tokens = self.proj(raw)
        if self.pos_emb is not None:
            L = tokens.shape[1]
            tokens = tokens + self.pos_emb(torch.arange(L, device=tokens.device))
        return tokens


class AutoSplitTokenizer(nn.Module):
    """Projects concatenated NS features into n_ns_tokens tokens."""

    def __init__(self, ns_input_dim: int, d_model: int, n_ns_tokens: int):
        super().__init__()
        out_dim = d_model * n_ns_tokens
        self.proj = nn.Sequential(
            nn.Linear(ns_input_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.n_ns_tokens = n_ns_tokens
        self.d_model = d_model

    def forward(self, ns_features: Tensor) -> Tensor:
        """ns_features: [B, ns_input_dim] -> [B, n_ns_tokens, d_model]"""
        return self.proj(ns_features).view(-1, self.n_ns_tokens, self.d_model)


class OneTransModel(BaseModel):
    """OneTrans: schema-driven Transformer for ranking.

    All architecture decisions are driven by ``Config``:

    - **Embedding tables**: from ``config.model.embedding_tables``
    - **Sequence groups**: from ``config.feature.sequence_groups`` — each group gets
      its own ``SequentialTokenizer``
    - **Scalar features**: from ``config.feature.scalar_features`` — concatenated
      into the NS-token input
    - **Dense features**: from ``config.feature.dense_features`` — projected to D
      (``project=True``, activation from ``df.activation``) or concatenated
      raw (``project=False``)
    - **Task heads**: from ``config.task_names``

    Pipeline mode (``_pipeline_mode=True``): forward() passes a pre-built KJT
    directly to ``emb.ec``, giving FX a traceable getitem→call_module path so
    TrainPipelineSparseDist can pipeline the embedding all-to-all.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device | None = None,
        meta_device: bool = False,
    ):
        super().__init__()
        self._pipeline_mode = False
        self.config = config
        mc = config.model
        fc = config.feature
        ot = mc.transformer

        if device is None:
            device = torch.device("cpu")
        emb_device = torch.device("meta") if meta_device else device
        self.tasks = list(config.task_names)
        D = mc.embedding_dim

        # Embeddings
        self.emb = TorchRecEmbeddings(
            mc.resolved_embedding_tables(),
            device=emb_device,
            embedding_init=mc.embedding_init,
            scalar_feature_names=set(fc.scalar_feature_names),
        )

        # Per-group sequential tokenizers
        self.seq_tokenizers = nn.ModuleDict({
            group_name: SequentialTokenizer(
                len(feats) * D, ot.d_model, max_len=2048, use_pos_embed=ot.pos_embed,
            ).to(device)
            for group_name, feats in fc.sequence_groups.items()
        })

        # Dense feature projections (project=True → Linear+activation, else raw concat)
        self.dense_projs = nn.ModuleDict()
        ns_raw_dim = len(fc.scalar_feature_names) * D
        for df in fc.dense_features:
            if df.project:
                act: nn.Module = nn.ReLU(inplace=True) if df.activation == "relu" else nn.GELU()
                self.dense_projs[df.name] = nn.Sequential(
                    nn.Linear(df.dim, D, device=device), act,
                )
                ns_raw_dim += D
            else:
                ns_raw_dim += df.dim

        self.ns_tokenizer = AutoSplitTokenizer(
            ns_raw_dim, ot.d_model, ot.n_ns_tokens,
        ).to(device)

        # Transformer stack
        self.blocks = nn.ModuleList([
            OneTransBlock(ot.d_model, ot.n_heads, ot.n_ns_tokens, ot.ffn_mult, ot.dropout,
                         attention_impl=ot.attention_impl)
            for _ in range(ot.n_layers)
        ]).to(device)
        self._compile_blocks = config.train.torch_compile
        self._compile_backend = config.train.torch_compile_backend
        self.final_norm = RMSNorm(ot.d_model).to(device)

        # Head bridge
        head_input_dim = ot.n_ns_tokens * ot.d_model
        self.head_proj = nn.Sequential(
            nn.Linear(head_input_dim, ot.d_model, device=device),
            nn.GELU(),
            nn.Dropout(ot.dropout),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(ot.d_model, 1, device=device)
            for task in self.tasks
        })

        # Contrastive heads: item representation uses only item-side scalars
        # (excludes user_id_feature) to avoid leaking user identity
        self._item_scalar_names = [sf.name for sf in fc.item_scalar_features]
        self.contrastive_user_proj = nn.Linear(ot.d_model, ot.d_model, device=device)
        self.contrastive_item_proj = nn.Sequential(
            nn.Linear(len(self._item_scalar_names) * D, ot.d_model, device=device),
            nn.ReLU(inplace=True),
        )

    def apply_compile(self) -> None:
        """Compile transformer blocks with torch.compile (call after DMP wrapping)."""
        if not self._compile_blocks:
            return
        logging.getLogger(__name__).info(
            f"Compiling {len(self.blocks)} OneTransBlocks with backend={self._compile_backend}")
        for i, block in enumerate(self.blocks):
            self.blocks[i] = torch.compile(
                block, fullgraph=False, dynamic=True, backend=self._compile_backend)

    def get_num_flops_per_sample(self) -> float:
        """Estimate FLOPs per sample for fwd+bwd (×3 matmul, ×2 FMA).

        Follows TorchTitan convention: embedding lookups excluded (memory-bound),
        all linear/einsum/attention ops counted with factor 6 for matmuls
        (3× fwd+bwd, 2× FMA) and 12 for attention (6× matmul directions, 2× FMA).
        Causal attention sparsity is not counted (same as PaLM/TorchTitan).
        """
        ot = self.config.model.transformer
        fc = self.config.feature
        D = ot.d_model
        H = ot.n_heads
        d_head = D // H
        n_ns = ot.n_ns_tokens
        ffn_dim = D * ot.ffn_mult
        emb_dim = self.config.model.embedding_dim

        L_hist = self.config.data.history_length
        n_groups = len(fc.sequence_groups)
        L_S = L_hist * n_groups
        schedule = pyramid_schedule(L_S, n_ns, ot.n_layers) if ot.use_pyramid else [L_S] * ot.n_layers

        flops = 0

        # Sequential tokenizers: Linear(raw_dim, D) + Linear(D, D) per group
        for group_name, feats in fc.sequence_groups.items():
            raw_dim = len(feats) * emb_dim
            flops += 6 * L_hist * (raw_dim * D + D * D)

        # Dense feature projections
        for df in fc.dense_features:
            if df.project:
                flops += 6 * df.dim * emb_dim

        # NS tokenizer: Linear(ns_raw_dim, D*n_ns) + Linear(D*n_ns, D*n_ns)
        ns_raw_dim = getattr(self.ns_tokenizer.proj[0], 'in_features', 0)
        ns_out_dim = n_ns * D
        if ns_raw_dim > 0:
            flops += 6 * (ns_raw_dim * ns_out_dim + ns_out_dim * ns_out_dim)

        # Transformer blocks (per layer with pyramid schedule)
        kv_len = L_S + n_ns
        for layer_idx in range(ot.n_layers):
            q_s = schedule[layer_idx]
            q_len = q_s + n_ns
            cur_n_s = kv_len - n_ns

            # S-token QKV: Linear(D, 3D) on cur_n_s tokens
            flops += 6 * cur_n_s * D * 3 * D
            # NS-token QKV: einsum [n_ns, 3D, D]
            flops += 6 * n_ns * 3 * D * D
            # Causal attention: Q*K^T + attn*V (full, not discounting causal sparsity)
            flops += 12 * H * d_head * q_len * kv_len
            # Output projection: Linear(D, D) on q_len tokens
            flops += 6 * q_len * D * D

            # FFN: S-tokens
            n_s_out = q_len - n_ns
            flops += 6 * n_s_out * D * ffn_dim * 2
            # FFN: NS-tokens (two einsums)
            flops += 6 * n_ns * D * ffn_dim * 2

            kv_len = q_len

        # Head projection: Linear(n_ns*D, D)
        flops += 6 * n_ns * D * D
        # Task heads: Linear(D, 1) per task
        flops += 6 * len(self.tasks) * D

        # Contrastive heads (only when contrastive_weight > 0)
        if self.config.train.contrastive_weight > 0:
            # contrastive_user_proj: Linear(D, D)
            flops += 6 * D * D
            # contrastive_item_proj: Linear(n_item_scalars*emb_dim, D)
            n_item_scalars = len(self._item_scalar_names)
            flops += 6 * n_item_scalars * emb_dim * D
            # cross_scores: user_emb @ item_emb.T  [B, D] @ [D, B] -> [B, B]
            # per-sample amortized: 2 * D * B (fwd only, ×3 for bwd = 6)
            batch_size = self.config.train.batch_size // (
                self.config.distributed.num_nodes * self.config.distributed.gpus_per_node)
            flops += 6 * D * batch_size

        return float(flops)

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def _lookup_all(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Schema-driven TorchRec lookup: maps batch keys to EC feature names."""
        feat_to_key = self.config.feature_to_batch_key()
        lookup_dict = {}
        for feat_name in self.config.feature.all_ec_feature_names():
            batch_key = feat_to_key.get(feat_name, feat_name)
            if batch_key in batch:
                lookup_dict[feat_name] = batch[batch_key]
        return self.emb(lookup_dict)

    def _lookup_pipeline(self, batch: dict) -> dict[str, Tensor]:
        """Embedding lookup via pre-built KJT (pipeline mode).

        Gives FX a clean ``batch["unpooled_kjt"] → emb.ec`` path so that
        TrainPipelineSparseDist can pipeline the embedding all-to-all.
        """
        ec_out = self.emb.ec(batch["unpooled_kjt"])
        first_ec_feat = self.config.feature.scalar_feature_names[0]
        first_batch_key = self.config.feature_to_batch_key().get(first_ec_feat, first_ec_feat)
        B = batch[first_batch_key].shape[0]
        embs: dict[str, Tensor] = {}
        for feat in self.emb._unpooled_features:
            jt = ec_out[feat]
            emb = jt.values()
            D = self.emb._feature_dims[feat]
            if feat in self.emb._scalar_features:
                embs[feat] = emb.view(B, D)
            else:
                embs[feat] = emb.view(B, -1, D)
        return embs

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def _build_pool_raw(
        self, embs: dict[str, Tensor], group_name: str,
    ) -> Tensor:
        """Concatenate embeddings for one sequence group: [B, L, N*D]."""
        feats = self.config.feature.sequence_groups[group_name]
        return torch.cat([embs[f] for f in feats], dim=-1)

    def _build_ns_raw(
        self, embs: dict[str, Tensor], batch: dict[str, Tensor],
    ) -> Tensor:
        """Build NS raw vector: scalar embeddings + dense features."""
        parts = [embs[f] for f in self.config.feature.scalar_feature_names]
        for df in self.config.feature.dense_features:
            if df.name in self.dense_projs:
                parts.append(self.dense_projs[df.name](batch[df.name]))
            else:
                parts.append(batch[df.name])
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def _backbone(
        self, embs: dict[str, Tensor], batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Run the full transformer and return (h, s_repr).

        h:      [B, d_model] head-projected NS-token representation.
        s_repr: [B, d_model] mean-pooled S-token representation (user history
                only, no candidate item info -- used for contrastive loss).
        """
        ot = self.config.model.transformer
        first_ec = self.config.feature.scalar_feature_names[0]
        first_key = self.config.feature_to_batch_key().get(first_ec, first_ec)
        B = batch[first_key].shape[0]
        L_NS = ot.n_ns_tokens

        # S-token construction: per-group tokenization
        pool_tokens = []
        for group_name in self.config.feature.sequence_groups:
            raw = self._build_pool_raw(embs, group_name)
            pool_tokens.append(self.seq_tokenizers[group_name](raw))
        s_tokens = torch.cat(pool_tokens, dim=1)
        L_S = s_tokens.shape[1]

        # NS-token construction
        ns_raw = self._build_ns_raw(embs, batch)
        ns_tokens = self.ns_tokenizer(ns_raw)

        # Concatenate: [S-tokens ; NS-tokens]
        x = torch.cat([s_tokens, ns_tokens], dim=1)

        # Pyramid schedule
        if ot.use_pyramid:
            schedule = pyramid_schedule(L_S, L_NS, ot.n_layers)
        else:
            schedule = [L_S] * ot.n_layers

        # Transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            q_s = schedule[layer_idx]
            q_len = q_s + L_NS
            kv_len = x.shape[1]
            cur_n_s = kv_len - L_NS

            mask = build_pyramid_mask(q_len, kv_len, x.device)
            x = block(x, mask=mask, q_len=q_len, n_s_tokens=cur_n_s)

        x = self.final_norm(x)

        # S-token user representation
        n_s_final = x.shape[1] - L_NS
        s_repr = x[:, :n_s_final].mean(dim=1)

        # NS-token head projection
        ns_out = x[:, -L_NS:]
        h = self.head_proj(ns_out.reshape(B, -1))

        return h, s_repr

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if self._pipeline_mode:  # concrete bool; FX traces only the active branch
            embs = self._lookup_pipeline(batch)
        else:
            embs = self._lookup_all(batch)
        h, s_repr = self._backbone(embs, batch)
        preds = {task: head(h).squeeze(-1) for task, head in self.heads.items()}
        return preds

    def forward_with_cross_scores(
        self, batch: dict[str, Tensor], cross_task: str = "",
    ) -> tuple[dict[str, Tensor], Tensor]:
        embs = self._lookup_all(batch)
        h, s_repr = self._backbone(embs, batch)
        preds = {task: head(h).squeeze(-1) for task, head in self.heads.items()}

        user_emb = self.contrastive_user_proj(s_repr)
        item_raw = torch.cat([embs[f] for f in self._item_scalar_names], dim=-1)
        item_emb = self.contrastive_item_proj(item_raw)

        cross_scores = user_emb @ item_emb.T
        return preds, cross_scores
