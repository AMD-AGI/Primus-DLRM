"""Distributed model wrapping: DDP/FSDP for dense layers, DMP for embeddings.

Supports three strategies:
  - ``"ddp"``  -- DistributedDataParallel (default, full replication)
  - ``"fsdp"`` -- FullyShardedDataParallel (stress testing, full sharding)
  - ``"dmp"``  -- TorchRec DistributedModelParallel for embedding sharding
                  with built-in DDP for dense layers
"""
from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from primus_dlrm.distributed.setup import get_local_rank, get_world_size

logger = logging.getLogger(__name__)


def wrap_model(
    model: nn.Module,
    device: torch.device,
    dense_strategy: Literal["ddp", "fsdp", "dmp"] = "ddp",
    embedding_sharding: str = "auto",
    embedding_lr: float = 1e-2,
    embedding_weight_decay: float = 1e-5,
    embedding_optimizer: str = "adam",
    embedding_eps: float = 1e-8,
) -> nn.Module:
    """Wrap a model for distributed training.

    Args:
        model: Model to wrap. For DMP, embedding modules must be on meta device.
        device: CUDA device for this rank.
        dense_strategy:
            ``"ddp"``  -- Full model replication via DDP (default).
            ``"fsdp"`` -- Full model sharding via FSDP (stress test).
            ``"dmp"``  -- TorchRec DMP for embedding sharding + DDP for dense.
        embedding_sharding: Sharding strategy for DMP. One of
            ``"auto"``, ``"table_wise"``, ``"row_wise"``, ``"data_parallel"``.
            Only used when ``dense_strategy="dmp"``.
        embedding_lr: Learning rate for the fused TBE embedding optimizer (DMP only).
        embedding_weight_decay: Weight decay for the fused TBE embedding optimizer.

    Returns:
        Wrapped model ready for distributed training.
    """
    world_size = get_world_size()

    if world_size <= 1 and dense_strategy != "dmp":
        logger.info("Single-GPU mode, skipping distributed wrapping.")
        return model.to(device)

    if dense_strategy == "ddp":
        model = model.to(device)
        model = DDP(model, device_ids=[get_local_rank()], find_unused_parameters=True)
        logger.info(f"Wrapped with DDP for {world_size} GPUs.")

    elif dense_strategy == "fsdp":
        model = model.to(device)
        model = FSDP(model, device_id=get_local_rank(), use_orig_params=True)
        logger.info(f"Wrapped with FSDP for {world_size} GPUs.")

    elif dense_strategy == "dmp":
        model = _wrap_dmp(model, device, embedding_sharding,
                          embedding_lr, embedding_weight_decay,
                          embedding_optimizer, embedding_eps)

    else:
        raise ValueError(f"Unknown dense_strategy: {dense_strategy!r}")

    return model


_SHARDING_MAP = {
    "auto": None,
    "table_wise": "table_wise",
    "row_wise": "row_wise",
    "data_parallel": "data_parallel",
    "column_wise": "column_wise",
}


def _wrap_dmp(
    model: nn.Module,
    device: torch.device,
    embedding_sharding: str,
    embedding_lr: float = 1e-2,
    embedding_weight_decay: float = 1e-5,
    embedding_optimizer: str = "adam",
    embedding_eps: float = 1e-8,
) -> nn.Module:
    """Wrap model with TorchRec DistributedModelParallel.

    DMP shards EBC/EC submodules across GPUs via FBGEMM TBE kernels.
    Dense submodules are automatically wrapped with DDP by DMP.
    Embedding optimizer is configurable: "adam" or "row_wise_adagrad".
    """
    from torch.distributed.optim import _apply_optimizer_in_backward
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.embedding import EmbeddingCollectionSharder
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.types import ShardingType
    from torchrec import EmbeddingBagCollection, EmbeddingCollection

    world_size = get_world_size()

    sharders = []
    has_ebc = any(isinstance(m, EmbeddingBagCollection) for m in model.modules())
    has_ec = any(isinstance(m, EmbeddingCollection) for m in model.modules())
    if has_ebc:
        sharders.append(EmbeddingBagCollectionSharder())
    if has_ec:
        sharders.append(EmbeddingCollectionSharder())

    if not sharders:
        logger.warning("No EBC/EC found in model — falling back to DDP.")
        model = model.to(device)
        return DDP(model, device_ids=[get_local_rank()], find_unused_parameters=True)

    if embedding_optimizer == "row_wise_adagrad":
        import torchrec.optim
        from fbgemm_gpu.split_table_batched_embeddings_ops import WeightDecayMode
        optim_cls = torchrec.optim.RowWiseAdagrad
        optim_kwargs = {
            "lr": embedding_lr, "eps": embedding_eps,
            "weight_decay": embedding_weight_decay,
            "weight_decay_mode": WeightDecayMode.L2,
        }
    else:
        optim_cls = torch.optim.Adam
        optim_kwargs = {
            "lr": embedding_lr, "eps": embedding_eps,
            "weight_decay": embedding_weight_decay,
        }

    n_emb_params = 0
    for m in model.modules():
        if isinstance(m, (EmbeddingBagCollection, EmbeddingCollection)):
            for p in m.parameters():
                _apply_optimizer_in_backward(optim_cls, [p], optim_kwargs)
                n_emb_params += 1
    logger.info(f"Applied fused {embedding_optimizer} to {n_emb_params} embedding params "
                f"(lr={embedding_lr}, eps={embedding_eps}, wd={embedding_weight_decay})")

    constraints = _build_constraints(model, embedding_sharding)

    if world_size <= 1:
        logger.info("Single-GPU DMP: placing all tables on GPU 0")
        dmp = DistributedModelParallel(module=model, device=device)
        logger.info(f"Wrapped with TorchRec DMP for 1 GPU.")
        return dmp

    topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints if constraints else None,
    )
    plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)
    _log_sharding_plan(plan)

    # Multi-node workaround: TorchRec's DMP uses global device IDs in shard
    # metadata (e.g. cuda:8) but tensors live on local devices (e.g. cuda:0
    # on node 2). PyTorch's ShardedTensor has strict device equality checks
    # that fail in multi-node. No upstream fix exists (see PyTorch #73096,
    # TorchRec #1739). We temporarily relax the checks during DMP construction,
    # but only when the global and local indices are consistent modulo the
    # number of GPUs per host.
    _GPUS_PER_HOST = 8

    from torch.distributed._shard.sharded_tensor.shard import Shard as _Shard
    from torch.distributed._shard.sharded_tensor import ShardedTensor as _ST

    _orig_post_init = _Shard.__post_init__
    _orig_init_from = _ST._init_from_local_shards_and_global_metadata

    def _is_same_local_gpu(dev_a, dev_b):
        """True if both are cuda and map to the same local GPU (index mod GPUS_PER_HOST)."""
        return (dev_a.type == "cuda" and dev_b.type == "cuda"
                and dev_a.index % _GPUS_PER_HOST == dev_b.index % _GPUS_PER_HOST)

    def _relaxed_post_init(self):
        tensor_dev = self.tensor.device
        placement_dev = self.metadata.placement.device()
        if tensor_dev == placement_dev:
            return
        if _is_same_local_gpu(tensor_dev, placement_dev):
            return
        raise ValueError(
            f"Local shard tensor device ({tensor_dev}) does not match "
            f"placement device ({placement_dev}) and they do not map to "
            f"the same local GPU (mod {_GPUS_PER_HOST})")

    @staticmethod
    def _patched_init_from(local_shards, sharded_tensor_metadata, *args, **kwargs):
        for shard in local_shards:
            placement = shard.metadata.placement
            if placement is not None:
                local_dev = shard.tensor.device
                placement_dev = placement.device()
                if local_dev != placement_dev and _is_same_local_gpu(local_dev, placement_dev):
                    from torch.distributed._shard.metadata import ShardMetadata
                    shard.metadata = ShardMetadata(
                        shard_offsets=shard.metadata.shard_offsets,
                        shard_sizes=shard.metadata.shard_sizes,
                        placement=f"rank:{placement.rank()}/{local_dev}",
                    )
        return _orig_init_from(local_shards, sharded_tensor_metadata, *args, **kwargs)

    _Shard.__post_init__ = _relaxed_post_init
    _ST._init_from_local_shards_and_global_metadata = _patched_init_from
    try:
        dmp = DistributedModelParallel(
            module=model,
            device=device,
            sharders=sharders,
            plan=plan,
        )
    finally:
        _Shard.__post_init__ = _orig_post_init
        _ST._init_from_local_shards_and_global_metadata = _orig_init_from

    logger.info(f"Wrapped with TorchRec DMP for {world_size} GPUs "
                f"(embedding_sharding={embedding_sharding}).")
    return dmp


def _build_constraints(
    model: nn.Module,
    embedding_sharding: str,
) -> dict[str, ParameterConstraints] | None:
    """Build per-table sharding constraints from user config."""
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.types import ShardingType
    from torchrec import EmbeddingBagCollection, EmbeddingCollection

    if embedding_sharding == "auto":
        return None

    stype_map = {
        "table_wise": ShardingType.TABLE_WISE,
        "row_wise": ShardingType.ROW_WISE,
        "data_parallel": ShardingType.DATA_PARALLEL,
        "column_wise": ShardingType.COLUMN_WISE,
    }
    stype = stype_map.get(embedding_sharding)
    if stype is None:
        logger.warning(f"Unknown sharding strategy {embedding_sharding!r}, using auto.")
        return None

    constraints = {}
    for mod in model.modules():
        if isinstance(mod, EmbeddingBagCollection):
            for cfg in mod.embedding_bag_configs():
                constraints[cfg.name] = ParameterConstraints(
                    sharding_types=[stype.value],
                )
        elif isinstance(mod, EmbeddingCollection):
            for cfg in mod.embedding_configs():
                constraints[cfg.name] = ParameterConstraints(
                    sharding_types=[stype.value],
                )

    return constraints if constraints else None


def _log_sharding_plan(plan) -> None:
    """Log the sharding plan produced by the planner."""
    if not plan or not plan.plan:
        return
    for module_fqn, param_plans in plan.plan.items():
        for param_name, param_sharding in param_plans.items():
            logger.info(f"  Shard: {module_fqn}.{param_name} -> "
                        f"type={param_sharding.sharding_type}, "
                        f"compute_kernel={param_sharding.compute_kernel}")


def is_dmp(model: nn.Module) -> bool:
    """Check if a model is wrapped with TorchRec DistributedModelParallel."""
    from torchrec.distributed import DistributedModelParallel
    return isinstance(model, DistributedModelParallel)
