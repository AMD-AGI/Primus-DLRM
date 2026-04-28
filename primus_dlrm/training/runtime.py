"""Runtime configuration for precision settings and CLI overrides."""
from __future__ import annotations

import logging

import torch

from primus_dlrm.config import Config, TrainConfig

logger = logging.getLogger(__name__)


def configure_runtime(tc: TrainConfig) -> None:
    """Set global precision flags based on training config.

    Call once at startup, before any computation. Configures:
    - BF16/TF32 precision modes
    """
    torch.backends.cuda.matmul.allow_tf32 = tc.allow_tf32
    torch.backends.cudnn.allow_tf32 = tc.allow_tf32

    logger.info(
      f"{torch.backends.cuda.matmul.allow_tf32=}, \
        {torch.backends.cudnn.allow_tf32=}"
    )


def apply_cli_overrides(config: Config, args) -> Config:
    """Apply CLI argument overrides to the Config object.

    This merges CLI flags into the config so downstream code only needs
    the config object, not both args and config.
    """
    overrides = []
    if hasattr(args, "dense_strategy"):
        config.distributed.dense_strategy = args.dense_strategy
        overrides.append(f"dense_strategy={args.dense_strategy}")
    if hasattr(args, "embedding_sharding"):
        config.distributed.embedding_sharding.strategy = args.embedding_sharding
        overrides.append(f"embedding_sharding={args.embedding_sharding}")
    if getattr(args, "attention_impl", None):
        config.model.transformer.attention_impl = args.attention_impl
        overrides.append(f"attention_impl={args.attention_impl}")
    if overrides:
        logger.info(f"CLI overrides applied: {', '.join(overrides)}")
    return config
