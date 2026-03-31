"""Synthetic random data generator for benchmarking and NaN reproduction.

Produces batches matching Config without requiring any real dataset.
Feature names, table sizes, dimensions, and value ranges are all configurable
via ``SyntheticDataConfig`` in the YAML.

Three usage patterns:

1. ``SyntheticDataset`` — standard ``Dataset`` returning per-sample dicts,
   compatible with DataLoader + collate.  Good for realistic DataLoader
   behavior with multiple workers.

2. ``generate_batch`` — bulk tensor generation (one call → full batch).
   Bypasses per-sample overhead for maximum throughput benchmarking.

3. ``SyntheticDataPipe`` — infinite iterator cycling through pre-generated
   batches (like the MI350X NaN reproducer's DataPipe).
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from torchrec import KeyedJaggedTensor
from torchrec.streamable import Pipelineable

from primus_dlrm.config import Config, SyntheticDataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Generates random samples matching Config.

    When ``num_samples > 0``, behaves as a finite map-style Dataset.
    When ``num_samples <= 0``, behaves as an infinite dataset.
    """

    def __init__(self, config: Config):
        self.config = config
        syn = config.data.synthetic
        self._syn = syn
        self._infinite = syn.num_samples <= 0
        self._table_sizes: dict[str, int] = {}
        for table in config.model.embedding_tables:
            for feat in table.features:
                self._table_sizes[feat] = table.num_embeddings

    def __len__(self) -> int:
        if self._infinite:
            return 10_000_000
        return self._syn.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = torch.Generator().manual_seed(self._syn.seed + idx)
        fc = self.config.feature
        feat_to_key = self.config.feature_to_batch_key()
        sample: dict[str, torch.Tensor] = {}

        for feat in fc.scalar_features:
            vocab = self._table_sizes[feat]
            key = feat_to_key.get(feat, feat)
            sample[key] = torch.randint(0, vocab, (1,), generator=rng).squeeze(0)

        for group_feats in fc.sequence_groups.values():
            for feat in group_feats:
                vocab = self._table_sizes[feat]
                key = feat_to_key.get(feat, feat)
                sample[key] = torch.randint(
                    0, vocab, (self.config.data.history_length,), generator=rng,
                )

        for df in fc.dense_features:
            lo, hi = df.value_range_min, df.value_range_max
            sample[df.name] = torch.empty(df.dim).uniform_(lo, hi)

        for task in self.config.task_names:
            sample[task] = torch.tensor(
                1.0 if torch.rand(1, generator=rng).item() < self._syn.label_positive_rate else 0.0,
            )

        return sample


def collate_synthetic(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack per-sample dicts into batched tensors."""
    keys = batch[0].keys()
    result: dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if vals[0].dim() == 0:
            result[k] = torch.stack(vals)
        else:
            result[k] = torch.stack(vals)
    return result


# ---------------------------------------------------------------------------
# Pipelineable batch for TorchRec pipeline mode
# ---------------------------------------------------------------------------

@dataclass
class SyntheticBatch(Pipelineable):
    """Generic batch for pipeline mode, wrapping a tensor dict + optional KJT."""
    tensors: dict[str, torch.Tensor]
    unpooled_kjt: Optional[KeyedJaggedTensor] = None

    def to(self, device: torch.device, non_blocking: bool = False) -> "SyntheticBatch":
        moved = {
            k: v.to(device=device, non_blocking=non_blocking)
            for k, v in self.tensors.items()
        }
        kjt = self.unpooled_kjt
        if kjt is not None:
            kjt = kjt.to(device=device, non_blocking=non_blocking)
        return SyntheticBatch(tensors=moved, unpooled_kjt=kjt)

    def record_stream(self, stream: torch.Stream) -> None:
        for v in self.tensors.values():
            if v.is_cuda:
                v.record_stream(stream)
        if self.unpooled_kjt is not None:
            self.unpooled_kjt.record_stream(stream)

    def pin_memory(self) -> "SyntheticBatch":
        pinned = {k: v.pin_memory() for k, v in self.tensors.items()}
        kjt = self.unpooled_kjt
        if kjt is not None:
            kjt = kjt.pin_memory()
        return SyntheticBatch(tensors=pinned, unpooled_kjt=kjt)

    def __getitem__(self, key: str) -> Any:
        if key == "unpooled_kjt":
            return self.unpooled_kjt
        return self.tensors[key]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.tensors)
        if self.unpooled_kjt is not None:
            d["unpooled_kjt"] = self.unpooled_kjt
        return d


def _build_synthetic_kjt(
    tensors: dict[str, torch.Tensor],
    config: Config,
) -> KeyedJaggedTensor:
    """Build KJT from all EC features."""
    feat_order = config.data.schema.kjt_feature_order or config.feature.all_ec_feature_names()
    feat_to_key = config.feature_to_batch_key()
    keys, all_values, all_lengths = [], [], []
    for feat in feat_order:
        batch_key = feat_to_key.get(feat, feat)
        if batch_key not in tensors:
            continue
        ids = tensors[batch_key]
        B = ids.shape[0]
        flat = ids.reshape(-1)
        keys.append(feat)
        all_values.append(flat)
        all_lengths.append(
            torch.full((B,), flat.shape[0] // B, dtype=torch.int32, device=ids.device),
        )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.cat(all_values) if all_values else torch.empty(0, dtype=torch.long),
        lengths=torch.cat(all_lengths) if all_lengths else torch.empty(0, dtype=torch.int32),
    )


def collate_synthetic_pipeline(
    batch: list[dict[str, torch.Tensor]],
    config: Config,
) -> SyntheticBatch:
    """Collate into a SyntheticBatch with pre-built KJT for pipeline mode."""
    tensors = collate_synthetic(batch)
    kjt = _build_synthetic_kjt(tensors, config)
    return SyntheticBatch(tensors=tensors, unpooled_kjt=kjt)


# ---------------------------------------------------------------------------
# Bulk batch generation (reproducer-style)
# ---------------------------------------------------------------------------

def generate_batch(
    config: Config,
    batch_size: int,
    seed: int = 42,
    label_positive_rate: float = 0.3,
    sparse_id_min: int = 0,
    sparse_id_max: int = 0,
    sparse_len_min: int = 0,
    sparse_len_max: int = 0,
) -> dict[str, torch.Tensor]:
    """Generate a full random batch directly as tensors."""
    rng = torch.Generator().manual_seed(seed)
    fc = config.feature
    table_sizes: dict[str, int] = {}
    for table in config.model.embedding_tables:
        for feat in table.features:
            table_sizes[feat] = table.num_embeddings

    feat_to_key = config.feature_to_batch_key()
    batch: dict[str, torch.Tensor] = {}
    B = batch_size
    L = config.data.history_length
    use_var_len = sparse_len_min > 0 or sparse_len_max > 0

    for feat in fc.scalar_features:
        id_max = sparse_id_max if sparse_id_max > 0 else table_sizes[feat]
        key = feat_to_key.get(feat, feat)
        batch[key] = torch.randint(sparse_id_min, id_max, (B,), generator=rng)

    for group_feats in fc.sequence_groups.values():
        for feat in group_feats:
            id_max = sparse_id_max if sparse_id_max > 0 else table_sizes[feat]
            key = feat_to_key.get(feat, feat)
            if use_var_len:
                lens = torch.randint(sparse_len_min, sparse_len_max + 1, (B,), generator=rng, dtype=torch.int32)
                vals = torch.randint(sparse_id_min, id_max, (lens.sum().item(),), generator=rng, dtype=torch.int32)
                batch[key] = vals
                batch[key + "__lengths"] = lens
            else:
                batch[key] = torch.randint(sparse_id_min, id_max, (B, L), generator=rng)

    for df in fc.dense_features:
        lo, hi = df.value_range_min, df.value_range_max
        batch[df.name] = torch.empty(B, df.dim).uniform_(lo, hi)

    for task in config.task_names:
        batch[task] = (torch.rand(B, generator=rng) < label_positive_rate).float()

    return batch


# ---------------------------------------------------------------------------
# Infinite data pipe (reproducer-style)
# ---------------------------------------------------------------------------

class SyntheticDataPipe:
    """Infinite iterator that produces synthetic batches.

    Two modes controlled by ``num_prebatched``:

    - **num_prebatched > 0** (reproducer-style): pre-generates N batches and
      cycles through them with ``deepcopy``.  Faster per step but repeats data.
    - **num_prebatched == 0**: generates a fresh random batch on every call.
      Slower but every batch is unique.
    """

    def __init__(
        self,
        config: Config,
        batch_size: int,
        num_prebatched: int = 16,
        seed: int = 42,
        label_positive_rate: float = 0.3,
        sparse_id_min: int = 0,
        sparse_id_max: int = 0,
        sparse_len_min: int = 0,
        sparse_len_max: int = 0,
    ):
        self._config = config
        self._batch_size = batch_size
        self._seed = seed
        self._label_positive_rate = label_positive_rate
        self._sparse_kwargs = dict(
            sparse_id_min=sparse_id_min, sparse_id_max=sparse_id_max,
            sparse_len_min=sparse_len_min, sparse_len_max=sparse_len_max,
        )
        self._idx = 0
        self.sampler = None

        if num_prebatched > 0:
            logger.info(f"Pre-generating {num_prebatched} synthetic batches (B={batch_size})...")
            self._batches = [
                generate_batch(config, batch_size, seed=seed + i,
                               label_positive_rate=label_positive_rate,
                               **self._sparse_kwargs)
                for i in range(num_prebatched)
            ]
        else:
            logger.info(f"Synthetic data: generating fresh batches on the fly (B={batch_size})")
            self._batches = None

    def __len__(self):
        return 10_000_000

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        self._idx += 1
        if self._batches is not None:
            return copy.deepcopy(self._batches[(self._idx - 1) % len(self._batches)])
        return generate_batch(
            self._config, self._batch_size,
            seed=self._seed + self._idx,
            label_positive_rate=self._label_positive_rate,
            **self._sparse_kwargs,
        )
