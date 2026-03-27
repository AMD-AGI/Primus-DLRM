"""Pipelineable batch dataclass for TorchRec TrainPipelineSparseDist.

TrainPipelineSparseDist requires batches that implement the ``Pipelineable``
interface (to, record_stream, pin_memory) so the pipeline can:

  Stage 0 (memcpy stream):     pin_memory() in DataLoader → .to(device,
                                non_blocking=True) for async H2D transfer.
  Stage 1 (data_dist stream):  record_stream() to prevent premature reuse of
                                GPU memory while the all-to-all is in flight.
  Stage 2 (default stream):    The batch is consumed by the model forward.

DlrmBatch wraps all per-sample tensors (IDs, features, labels) into a single
Pipelineable object and pre-builds a KeyedJaggedTensor (KJT) so the pipeline's
FX tracer can discover the ``batch["unpooled_kjt"] → EmbeddingCollection``
path for scheduling input/output distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import torch
from torchrec import KeyedJaggedTensor
from torchrec.streamable import Pipelineable

from primus_dlrm.data.dataset import ScoringPair

# NOTE: This mapping and the DlrmBatch fields below are specific to the
# Yambda dataset schema (ScoringPair).  For generic/synthetic data, use
# SyntheticBatch from data/synthetic.py instead.  The model itself is
# feature-name agnostic — it reads feature names from the FeatureSchema.
#
# Maps each raw ID field in the batch to the corresponding feature name in the
# EmbeddingCollection.  Order must match TorchRecEmbeddings._unpooled_features
# so the KJT keys align with EC's feature-to-table mapping.
#
# There are 13 features across 4 embedding tables:
#   item  table: item_id + 3 history channels (lp/like/skip)
#   artist table: artist_id + 3 history channels
#   album table: album_id + 3 history channels
#   uid   table: uid (single feature, no history)
_BATCH_TO_EC: list[tuple[str, str]] = [
    ("item_id",              "item"),
    ("artist_id",            "artist"),
    ("album_id",             "album"),
    ("hist_lp_item_ids",     "hist_lp_item"),
    ("hist_like_item_ids",   "hist_like_item"),
    ("hist_skip_item_ids",   "hist_skip_item"),
    ("hist_lp_artist_ids",   "hist_lp_artist"),
    ("hist_like_artist_ids", "hist_like_artist"),
    ("hist_skip_artist_ids", "hist_skip_artist"),
    ("hist_lp_album_ids",    "hist_lp_album"),
    ("hist_like_album_ids",  "hist_like_album"),
    ("hist_skip_album_ids",  "hist_skip_album"),
    ("uid",                  "uid"),
]


def _build_kjt(tensors: dict[str, torch.Tensor]) -> KeyedJaggedTensor:
    """Build a KeyedJaggedTensor from raw ID tensors for all EC features.

    Each ID tensor has shape (B,) for single-value features (item_id, uid) or
    (B, L) for sequence features (history).  They are flattened into the KJT's
    ``values`` buffer with per-sample ``lengths`` so EC can look up embeddings
    in a single fused kernel call.

    The KJT is built on CPU at collate time (before H2D), so the pipeline's
    memcpy stage transfers it to GPU along with the rest of the batch.
    """
    keys, all_values, all_lengths = [], [], []
    for batch_field, ec_name in _BATCH_TO_EC:
        ids = tensors[batch_field]
        B = ids.shape[0]
        flat = ids.reshape(-1)
        keys.append(ec_name)
        all_values.append(flat)
        # Uniform lengths: every sample has the same number of IDs per feature
        # (1 for single-value, L for history sequences).
        all_lengths.append(torch.full((B,), flat.shape[0] // B,
                                      dtype=torch.int32, device=ids.device))
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys, values=torch.cat(all_values), lengths=torch.cat(all_lengths),
    )


@dataclass
class DlrmBatch(Pipelineable):
    """Batch of scoring pairs implementing TorchRec's Pipelineable interface.

    Fields fall into three categories:

    1. **Embedding ID fields** — fed into the EmbeddingCollection via the
       pre-built ``unpooled_kjt``.  These drive the all-to-all redistribution
       in the pipeline's data_dist stage (``all2all_data:kjt splits/lengths/
       values``) and the output redistribution (``All2All_Seq_fwd``).

    2. **Dense feature fields** — ``audio_embed`` and optional counter tensors.
       Passed directly to the model's dense layers, no embedding lookup needed.

    3. **Label fields** — ``listen_plus``, ``like``, ``dislike``,
       ``listen_pct``.  Used by the loss function after the forward pass.
    """

    # --- Embedding ID fields: history sequences (B, L) ---
    hist_lp_item_ids: torch.Tensor
    hist_lp_artist_ids: torch.Tensor
    hist_lp_album_ids: torch.Tensor
    hist_like_item_ids: torch.Tensor
    hist_like_artist_ids: torch.Tensor
    hist_like_album_ids: torch.Tensor
    hist_skip_item_ids: torch.Tensor
    hist_skip_artist_ids: torch.Tensor
    hist_skip_album_ids: torch.Tensor

    # --- Embedding ID fields: single-value (B,) ---
    uid: torch.Tensor
    item_id: torch.Tensor
    artist_id: torch.Tensor
    album_id: torch.Tensor

    # --- Dense feature fields ---
    audio_embed: torch.Tensor           # (B, audio_embed_dim)

    # --- Label fields ---
    listen_plus: torch.Tensor           # (B,) binary
    like: torch.Tensor                  # (B,) binary
    dislike: torch.Tensor               # (B,) binary
    listen_pct: torch.Tensor            # (B,) continuous [0, 1]

    # --- Optional dense counter features ---
    user_counters: Optional[torch.Tensor] = None     # (B, n_counter_features)
    item_counters: Optional[torch.Tensor] = None     # (B, n_counter_features)
    cross_counters: Optional[torch.Tensor] = None    # (B, n_counter_features)

    # Pre-built KJT containing all 13 embedding features.  Built at collate
    # time so that the pipeline's FX tracer sees a static graph path:
    #   batch["unpooled_kjt"] → EmbeddingCollection(kjt)
    # Without this, the tracer would have to trace through _build_kjt which
    # creates tensors dynamically and is not FX-friendly.
    unpooled_kjt: Optional[KeyedJaggedTensor] = None

    def _apply_to_fields(self, fn):
        """Apply ``fn`` to every Tensor/KJT field, returning a kwargs dict."""
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (torch.Tensor, KeyedJaggedTensor)):
                kwargs[f.name] = fn(val)
            else:
                kwargs[f.name] = val
        return kwargs

    def to(self, device: torch.device, non_blocking: bool = False) -> "DlrmBatch":
        """Move all tensors to ``device``.  Called by the pipeline on the
        memcpy stream with non_blocking=True for async H2D transfer."""
        return DlrmBatch(**self._apply_to_fields(
            lambda v: v.to(device=device, non_blocking=non_blocking)))

    def record_stream(self, stream: torch.Stream) -> None:
        """Mark all GPU tensors as used by ``stream`` so the caching allocator
        won't reclaim their memory until ``stream`` has consumed them.  Called
        by the pipeline after the async H2D copy to protect tensors that will
        be read on the data_dist or default stream."""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, KeyedJaggedTensor):
                val.record_stream(stream)
            elif isinstance(val, torch.Tensor) and val.is_cuda:
                val.record_stream(stream)

    def pin_memory(self) -> "DlrmBatch":
        """Pin all tensors into page-locked host memory.  Required for
        non_blocking=True H2D transfers.  Called automatically when the
        DataLoader has pin_memory=True."""
        return DlrmBatch(**self._apply_to_fields(lambda v: v.pin_memory()))

    def __getitem__(self, key: str) -> Any:
        """Attribute access by string key.  Required by TorchRec's ArgInfo
        so the FX-traced graph can do ``batch["unpooled_kjt"]`` to extract
        the KJT at runtime."""
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict, dropping None fields.  Used by
        PipelineModelWrapper to pass the batch to the model's forward()
        which expects a dict, not a dataclass."""
        return {f.name: v for f in fields(self) if (v := getattr(self, f.name)) is not None}


def collate_pipeline_batch(batch: list[ScoringPair]) -> DlrmBatch:
    """Collate a list of ScoringPairs into a DlrmBatch with pre-built KJT.

    Used as the ``collate_fn`` for the DataLoader in pipeline mode.  Stacks
    individual samples into batched tensors and builds the KJT on CPU so it's
    ready for the pipeline's H2D transfer stage.
    """
    tensors: dict[str, torch.Tensor] = {
        "hist_lp_item_ids": torch.stack([b.hist_lp_item_ids for b in batch]),
        "hist_lp_artist_ids": torch.stack([b.hist_lp_artist_ids for b in batch]),
        "hist_lp_album_ids": torch.stack([b.hist_lp_album_ids for b in batch]),
        "hist_like_item_ids": torch.stack([b.hist_like_item_ids for b in batch]),
        "hist_like_artist_ids": torch.stack([b.hist_like_artist_ids for b in batch]),
        "hist_like_album_ids": torch.stack([b.hist_like_album_ids for b in batch]),
        "hist_skip_item_ids": torch.stack([b.hist_skip_item_ids for b in batch]),
        "hist_skip_artist_ids": torch.stack([b.hist_skip_artist_ids for b in batch]),
        "hist_skip_album_ids": torch.stack([b.hist_skip_album_ids for b in batch]),
        "uid": torch.tensor([b.uid for b in batch], dtype=torch.long),
        "item_id": torch.tensor([b.item_id for b in batch], dtype=torch.long),
        "artist_id": torch.tensor([b.artist_id for b in batch], dtype=torch.long),
        "album_id": torch.tensor([b.album_id for b in batch], dtype=torch.long),
        "audio_embed": torch.stack([b.audio_embed for b in batch]),
        "listen_plus": torch.tensor([b.listen_plus for b in batch], dtype=torch.float32),
        "like": torch.tensor([b.like for b in batch], dtype=torch.float32),
        "dislike": torch.tensor([b.dislike for b in batch], dtype=torch.float32),
        "listen_pct": torch.tensor([b.listen_pct for b in batch], dtype=torch.float32),
    }
    # Counter features are optional; only present when the dataset was
    # built with enable_counters=true.
    if batch[0].user_counters is not None:
        tensors["user_counters"] = torch.stack([b.user_counters for b in batch])
        tensors["item_counters"] = torch.stack([b.item_counters for b in batch])
        tensors["cross_counters"] = torch.stack([b.cross_counters for b in batch])
    return DlrmBatch(**tensors, unpooled_kjt=_build_kjt(tensors))
