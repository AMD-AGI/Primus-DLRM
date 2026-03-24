"""Runtime configuration for precision settings."""
from __future__ import annotations

import logging

import torch

from primus_dlrm.config import TrainConfig

logger = logging.getLogger(__name__)


def configure_runtime(tc: TrainConfig) -> None:
    """Set global precision flags based on training config.

    Call once at startup, before any computation. Configures:
    - BF16/TF32 precision modes
    """
    torch.backends.cuda.matmul.allow_tf32 = tc.allow_tf32
    torch.backends.cudnn.allow_tf32 = tc.allow_tf32

    precision = []
    if tc.bf16:
        precision.append("BF16 autocast")
    if tc.allow_tf32:
        precision.append("TF32 matmul")
    if not precision:
        precision.append("FP32")
    logger.info(f"Precision: {' + '.join(precision)}")
