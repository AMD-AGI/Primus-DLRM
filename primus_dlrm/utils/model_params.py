"""Parameter counting + storage-size estimation + startup logging.

Used by `dist_trainer` to print a per-component breakdown of dense and sparse
(embedding-bag) parameters, including dtype-aware byte sizes and per-rank vs
total memory footprint for both weights and optimizer state.

All functions are public (no leading underscore) so they can be reused by
other utilities (e.g., a standalone `print-model-summary` script).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "classify_dense_param",
    "count_embedding_params_from_config",
    "dtype_bytes",
    "fmt_bytes",
    "is_embedding_param",
    "log_param_summary",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype + formatting helpers
# ---------------------------------------------------------------------------

def dtype_bytes(dt: torch.dtype) -> int:
    """Bytes per element for a torch dtype."""
    if dt in (torch.float64,):
        return 8
    if dt in (torch.float32, torch.int32):
        return 4
    if dt in (torch.bfloat16, torch.float16, torch.int16):
        return 2
    if dt in (torch.int8, torch.uint8, torch.bool):
        return 1
    if dt in (torch.int64,):
        return 8
    return 4  # safe fallback


def fmt_bytes(b: float) -> str:
    """Human-readable bytes (GiB / MiB / KiB / B)."""
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KiB"
    return f"{int(b)} B"


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

def is_embedding_param(name: str) -> bool:
    """Heuristic to identify embedding parameters managed by DMP / TorchRec.

    Matches the standard naming convention used by TorchRec wrappers:
      - `*ebc.*`        — EmbeddingBagCollection (pooled)
      - `*ec.*`         — EmbeddingCollection (unpooled)
      - `*embedding*`   — defensive catch-all for any other embedding subtree

    NOTE: plain `nn.Embedding` modules with names like `pos_emb` are NOT
    matched (no "embedding" substring), so they're correctly counted as dense.
    """
    return any(tok in name for tok in ("ebc.", "ec.", "embedding"))


def classify_dense_param(name: str) -> str:
    """Bucket a dense parameter by top-level model component.

    Returns one of: 'transformer-blocks', 'tokenizers', 'dense-projections',
    'heads', 'contrastive', 'top-level norms', 'other'.

    The buckets follow the naming convention in `OneTrans` model:
      - `blocks.*`              → transformer blocks (S-/NS-attn, FFNs, in-block RMSNorms)
      - `seq_tokenizers.*`      → sequence tokenizer projections + position embeddings
      - `dense_projs.*`         → per-dense-feature input MLPs
      - `head_proj.*`           → final-token projection before per-task heads
      - `heads.*`               → per-task prediction heads
      - `contrastive_*_proj.*`  → contrastive loss user/item projections
      - `*norm*`                → top-level normalization layers
    """
    nl = name.lower()
    if 'blocks.' in nl:
        return 'transformer-blocks'
    if 'seq_tokenizer' in nl or 'tokenizer' in nl:
        return 'tokenizers'
    if 'dense_proj' in nl:
        return 'dense-projections'
    if 'head_proj' in nl or nl.startswith('heads.') or '.heads.' in nl:
        return 'heads'
    if 'contrastive' in nl:
        return 'contrastive'
    if 'norm' in nl:
        return 'top-level norms'
    return 'other'


# ---------------------------------------------------------------------------
# Embedding size from config
# ---------------------------------------------------------------------------

def count_embedding_params_from_config(
    config,
) -> tuple[int, int, list[tuple[str, int, int, str]]]:
    """Read embedding-table sizes from the resolved config.

    Returns ``(total_emb_params, total_emb_bytes, per_table_breakdown)`` where
    each row in the breakdown is ``(name, num_embeddings, embedding_dim, dtype_str)``.

    These are TOTAL (unsharded) counts — DMP shards rows across ranks, but the
    global model size is shape-config-driven.

    Why we use config rather than ``model.named_parameters()``:
      - DMP wraps `EmbeddingBagCollection` into `ShardedEmbeddingBagCollection`,
        whose underlying FBGEMM TBE keeps weights as **buffers**, not parameters
        — so they don't appear in ``named_parameters()`` at all.

    FBGEMM TBE default ``weights_precision`` on AMD ROCm is FP32 (overridden
    in `embedding.py` via cache_precision). We assume FP32 unless the table
    config carries a `dtype` attribute (forward-compatible with future bf16
    embedding support).
    """
    try:
        tables = config.model.resolved_embedding_tables()
    except Exception:
        return 0, 0, []
    rows: list[tuple[str, int, int, str]] = []
    total_params = 0
    total_bytes = 0
    for tbl in tables:
        dt = getattr(tbl, "dtype", torch.float32)
        if isinstance(dt, str):
            dt = getattr(torch, dt, torch.float32)
        n = tbl.num_embeddings * tbl.embedding_dim
        total_params += n
        total_bytes += n * dtype_bytes(dt)
        rows.append((tbl.name, tbl.num_embeddings, tbl.embedding_dim,
                     str(dt).replace("torch.", "")))
    return total_params, total_bytes, rows


# ---------------------------------------------------------------------------
# Top-level summary logger
# ---------------------------------------------------------------------------

def log_param_summary(
    model: nn.Module,
    dense_params: list[torch.Tensor],
    dense_optimizer: str,
    world_size: int,
    config: Optional[object] = None,
) -> None:
    """Print a detailed parameter-count + storage-size summary at startup.

    Reports (carefully accounting for dtype):
      - Dense params (replicated on each rank, model dtype) — also broken down
        by top-level component (transformer blocks vs tokenizers/heads/etc.)
      - Embedding params (sharded across ranks, FBGEMM TBE dtype, usually fp32)
      - Optimizer state (AdamW: m + v in fp32 per dense param; row-wise/global Adam
        state on TBE varies; we report standard Adam = 2× param bytes on dense).
      - Total model + optimizer memory footprint per rank and across cluster.
    """
    # --- Dense ---
    n_dense = sum(p.numel() for p in dense_params)
    if dense_params:
        from collections import Counter
        dtype_counts = Counter(p.dtype for p in dense_params)
        dense_dtype = dtype_counts.most_common(1)[0][0]
    else:
        dense_dtype = torch.float32
    dense_bytes_per = dtype_bytes(dense_dtype)
    dense_bytes = n_dense * dense_bytes_per

    # Per-component breakdown of dense params (audit aid)
    dense_by_component: dict[str, int] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or is_embedding_param(n):
            continue
        bucket = classify_dense_param(n)
        dense_by_component[bucket] = dense_by_component.get(bucket, 0) + p.numel()

    # --- Embedding (total, unsharded — read from config) ---
    n_emb_total, emb_bytes_total, table_rows = (
        count_embedding_params_from_config(config) if config is not None else (0, 0, [])
    )
    n_emb_per_rank = n_emb_total // max(1, world_size)
    emb_bytes_per_rank = emb_bytes_total // max(1, world_size)

    # --- Optimizer state (best-effort estimate; Adam = 2× param bytes in fp32) ---
    # Dense AdamW: m (fp32) + v (fp32) = 8 B/param regardless of param dtype
    dense_opt_bytes = n_dense * 8
    # TBE Adam: m + v in fp32 = 2× emb_bytes
    emb_opt_bytes_total = n_emb_total * 2 * 4
    emb_opt_bytes_per_rank = emb_opt_bytes_total // max(1, world_size)

    # --- Log ---
    logger.info("=" * 70)
    logger.info("Model parameter summary")
    logger.info("=" * 70)
    logger.info(f"  Dense params:  {n_dense:>15,d}  ({fmt_bytes(dense_bytes):>10s},"
                f" dtype={str(dense_dtype).replace('torch.', '')}, replicated on each rank)")
    if dense_by_component:
        ordered = sorted(dense_by_component.items(),
                         key=lambda x: (x[0] != 'transformer-blocks', -x[1]))
        for bucket, n in ordered:
            pct = n / n_dense * 100 if n_dense else 0
            n_bytes = n * dense_bytes_per
            logger.info(f"    {bucket:<22s}: {n:>14,d}  ({fmt_bytes(n_bytes):>10s},"
                        f" {pct:>5.1f}% of dense)")
    logger.info(f"  Sparse params: {n_emb_total:>15,d}  ({fmt_bytes(emb_bytes_total):>10s},"
                f" total — sharded {n_emb_per_rank:,}/rank ≈ {fmt_bytes(emb_bytes_per_rank)})")
    logger.info(f"  TOTAL params:  {n_dense + n_emb_total:>15,d}"
                f"  ({fmt_bytes(dense_bytes + emb_bytes_total)} total weights)")

    if table_rows:
        logger.info("  Embedding tables (rows × dim × dtype):")
        for name, n_rows, dim, dt_s in table_rows:
            tbl_bytes = n_rows * dim * dtype_bytes(getattr(torch, dt_s))
            logger.info(f"    {name:<10s}: {n_rows:>10,d} × {dim:>3d} × {dt_s:<8s}"
                        f" = {fmt_bytes(tbl_bytes):>10s}")

    logger.info(f"  Optimizer state ({dense_optimizer} on dense + Adam on TBE, all fp32 m+v):")
    logger.info(f"    Dense AdamW (m,v fp32):       {fmt_bytes(dense_opt_bytes)}/rank (replicated)")
    logger.info(f"    TBE Adam (m,v fp32):          {fmt_bytes(emb_opt_bytes_total)} total"
                f" → {fmt_bytes(emb_opt_bytes_per_rank)}/rank")
    logger.info(f"  Memory footprint per rank (weights + optimizer):")
    per_rank_total = dense_bytes + dense_opt_bytes + emb_bytes_per_rank + emb_opt_bytes_per_rank
    logger.info(f"    weights:  dense {fmt_bytes(dense_bytes)} + emb-shard {fmt_bytes(emb_bytes_per_rank)}"
                f" = {fmt_bytes(dense_bytes + emb_bytes_per_rank)}")
    logger.info(f"    opt state: dense {fmt_bytes(dense_opt_bytes)} + emb-shard {fmt_bytes(emb_opt_bytes_per_rank)}"
                f" = {fmt_bytes(dense_opt_bytes + emb_opt_bytes_per_rank)}")
    logger.info(f"    TOTAL per rank: {fmt_bytes(per_rank_total)}")
    logger.info("=" * 70)
