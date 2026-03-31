"""Schema-driven Pipelineable batch for TorchRec TrainPipelineSparseDist.

TrainPipelineSparseDist requires batches that implement the ``Pipelineable``
interface (to, record_stream, pin_memory) so the pipeline can:

  Stage 0 (memcpy stream):     pin_memory() in DataLoader → .to(device,
                                non_blocking=True) for async H2D transfer.
  Stage 1 (data_dist stream):  record_stream() to prevent premature reuse of
                                GPU memory while the all-to-all is in flight.
  Stage 2 (default stream):    The batch is consumed by the model forward.

``PipelineBatch`` is a generic dict-based Pipelineable that wraps any set of
tensors plus an optional KJT.  No hardcoded feature names — the KJT is built
from the Config at collate time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torchrec import KeyedJaggedTensor
from torchrec.streamable import Pipelineable

from primus_dlrm.config import Config


@dataclass
class PipelineBatch(Pipelineable):
    """Generic dict-based Pipelineable batch for any schema."""
    tensors: dict[str, torch.Tensor]
    unpooled_kjt: Optional[KeyedJaggedTensor] = None

    def to(self, device: torch.device, non_blocking: bool = False) -> "PipelineBatch":
        moved = {
            k: v.to(device=device, non_blocking=non_blocking)
            for k, v in self.tensors.items()
        }
        kjt = self.unpooled_kjt
        if kjt is not None:
            kjt = kjt.to(device=device, non_blocking=non_blocking)
        return PipelineBatch(tensors=moved, unpooled_kjt=kjt)

    def record_stream(self, stream: torch.Stream) -> None:
        for v in self.tensors.values():
            if v.is_cuda:
                v.record_stream(stream)
        if self.unpooled_kjt is not None:
            self.unpooled_kjt.record_stream(stream)

    def pin_memory(self) -> "PipelineBatch":
        pinned = {k: v.pin_memory() for k, v in self.tensors.items()}
        kjt = self.unpooled_kjt
        if kjt is not None:
            kjt = kjt.pin_memory()
        return PipelineBatch(tensors=pinned, unpooled_kjt=kjt)

    def __getitem__(self, key: str) -> Any:
        if key == "unpooled_kjt":
            return self.unpooled_kjt
        return self.tensors[key]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.tensors)
        if self.unpooled_kjt is not None:
            d["unpooled_kjt"] = self.unpooled_kjt
        return d


def build_kjt(
    tensors: dict[str, torch.Tensor],
    config: Config,
) -> KeyedJaggedTensor:
    """Build a KJT from batch tensors using the config's feature order."""
    feat_order = config.data.schema.kjt_feature_order or config.feature.all_ec_feature_names()
    feat_to_key = config.feature_to_batch_key()
    keys, all_values, all_lengths = [], [], []
    for feat in feat_order:
        batch_key = feat_to_key.get(feat, feat)
        if batch_key not in tensors:
            continue
        ids = tensors[batch_key]
        lengths_key = batch_key + "__lengths"
        if lengths_key in tensors:
            keys.append(feat)
            all_values.append(ids)
            all_lengths.append(tensors[lengths_key])
        else:
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


def collate_pipeline_batch(
    batch: list,
    config: Config,
) -> PipelineBatch:
    """Collate a list of sample dicts into a PipelineBatch with pre-built KJT.

    Each sample is a ``dict[str, Tensor]`` (from ``ScoringPair`` via
    ``collate_to_dict``, or from ``SyntheticDataset``).  Feature names
    and KJT construction are driven by the schema — no hardcoded names.
    """
    from primus_dlrm.data.dataset import ScoringPair

    # Convert ScoringPairs to dicts if needed
    if batch and isinstance(batch[0], ScoringPair):
        batch = [_scoring_pair_to_dict(b) for b in batch]

    # Stack per-sample dicts into batched tensors
    keys = batch[0].keys()
    tensors: dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            tensors[k] = torch.stack(vals) if vals[0].dim() > 0 else torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            dtype = torch.float32 if isinstance(vals[0], float) else torch.long
            tensors[k] = torch.tensor(vals, dtype=dtype)

    kjt = build_kjt(tensors, config)
    return PipelineBatch(tensors=tensors, unpooled_kjt=kjt)


def _scoring_pair_to_dict(sp) -> dict[str, Any]:
    """Convert a ScoringPair dataclass to a plain dict."""
    from dataclasses import fields as dc_fields
    return {f.name: getattr(sp, f.name) for f in dc_fields(sp) if getattr(sp, f.name) is not None}
