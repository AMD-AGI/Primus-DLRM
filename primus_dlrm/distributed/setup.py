"""Process group initialization and rank helpers for torchrun / srun."""
from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Works with both launch modes:
    - ``torchrun --rdzv_backend=c10d`` (sets RANK, WORLD_SIZE, LOCAL_RANK, etc.)
    - ``srun torchrun --master_addr/--master_port/--node_rank`` (same env vars)

    RCCL on ROCm is NCCL-API-compatible, so ``backend="nccl"`` works for both.
    """
    if dist.is_initialized():
        return

    # torchrun always sets these; if missing we're single-process
    if "RANK" not in os.environ:
        logger.info("No RANK env var -- running in single-process mode.")
        return

    from datetime import timedelta
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Distributed init: rank={get_rank()}/{get_world_size()}, "
        f"local_rank={local_rank}, backend={backend}"
    )


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
