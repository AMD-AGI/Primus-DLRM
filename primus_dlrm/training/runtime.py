"""Runtime configuration for precision and FBGEMM settings."""
from __future__ import annotations

import logging
import os

import torch

from primus_dlrm.config import TrainConfig

logger = logging.getLogger(__name__)


def configure_runtime(tc: TrainConfig) -> None:
    """Set global precision and runtime flags based on training config.

    Call once at startup, before any computation. Configures:
    - BF16/TF32 precision modes
    - FBGEMM TBE V2 kernels (embedding compute-communication overlap)
    """
    # Precision
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

    # FBGEMM TBE V2
    if tc.tbe_v2:
        os.environ["FBGEMM_TBE_V2"] = "1"
        logger.info("FBGEMM TBE V2 enabled")
