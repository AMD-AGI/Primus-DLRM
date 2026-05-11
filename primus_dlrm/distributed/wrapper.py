"""Distributed model wrapping: DDP/FSDP for dense layers, DMP for embeddings.

Supports three strategies:
  - ``"ddp"``  -- DistributedDataParallel (default, full replication)
  - ``"fsdp"`` -- FullyShardedDataParallel (stress testing, full sharding)
  - ``"dmp"``  -- TorchRec DistributedModelParallel for embedding sharding
                  with built-in DDP for dense layers
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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
    config,
) -> nn.Module:
    """Wrap a model for distributed training.

    All settings are read from ``config``:
    - ``config.distributed.dense_strategy``: ``"ddp"`` | ``"fsdp"`` | ``"dmp"``
    - ``config.distributed.embedding_sharding.strategy``: DMP sharding strategy
    - ``config.train``: embedding optimizer settings (lr, eps, etc.)

    Call ``apply_cli_overrides(config, args)`` before this to merge CLI flags.

    Args:
        model: Model to wrap. For DMP, embedding modules must be on meta device.
        device: CUDA device for this rank.
        config: Config object (use ``apply_cli_overrides`` to merge CLI args first).

    Returns:
        Wrapped model ready for distributed training.
    """
    dense_strategy = config.distributed.dense_strategy
    tc = config.train
    embedding_lr = tc.embedding_lr
    embedding_weight_decay = tc.weight_decay
    embedding_optimizer = tc.embedding_optimizer
    embedding_eps = tc.embedding_eps
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
        model = _wrap_dmp(model, device, config,
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

# Map our string compute_kernel values to TorchRec's enum strings. Keep the
# YAML user-facing names short and self-documenting; widen this dict if a new
# FBGEMM kernel becomes useful.
_COMPUTE_KERNEL_MAP = {
    "fused": "fused",
    "fused_uvm": "fused_uvm",
    "fused_uvm_caching": "fused_uvm_caching",
    "dense": "dense",
}

# Map output_dtype YAML strings to TorchRec DataType enum members. Lazily
# resolved inside ``_build_constraints`` so importing this module does not
# require torchrec at top level.
_OUTPUT_DTYPE_MAP = {
    "fp32": "FP32",
    "fp16": "FP16",
    "bf16": "BF16",
}


def _wrap_dmp(
    model: nn.Module,
    device: torch.device,
    config,
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

    embedding_sharding = config.distributed.embedding_sharding.strategy

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

    constraints = _build_constraints(config)

    if world_size <= 1:
        logger.info("Single-GPU DMP: placing all tables on GPU 0")
        dmp = DistributedModelParallel(module=model, device=device)
        logger.info(f"Wrapped with TorchRec DMP for 1 GPU.")
        return dmp

    topology_kwargs = _build_topology_kwargs(config, world_size)
    logger.info(f"TorchRec topology overrides: {topology_kwargs}")
    topology = Topology(**topology_kwargs)
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


_STYPE_MAP_FACTORY = None  # populated lazily once torchrec is importable


def _stype_map():
    """Cache the YAML-name -> ``ShardingType.value`` map (lazy torchrec import)."""
    global _STYPE_MAP_FACTORY
    if _STYPE_MAP_FACTORY is None:
        from torchrec.distributed.types import ShardingType
        _STYPE_MAP_FACTORY = {
            "table_wise":    ShardingType.TABLE_WISE.value,
            "row_wise":      ShardingType.ROW_WISE.value,
            "data_parallel": ShardingType.DATA_PARALLEL.value,
            "column_wise":   ShardingType.COLUMN_WISE.value,
        }
    return _STYPE_MAP_FACTORY


def _build_constraints(
    config,
) -> dict[str, "ParameterConstraints"] | None:
    """Build per-table TorchRec ``ParameterConstraints`` from the config.

    Per-field precedence (first set wins):

    1. ``EmbeddingTableConfig.sharding`` (inline per-table block)
    2. ``EmbeddingShardingConfig.default_table_sharding`` (cross-table default)
    3. ``EmbeddingShardingConfig.strategy`` (legacy global string; only
       contributes ``sharding_type``)

    Returns ``None`` when nothing ends up constrained, so the planner runs
    in fully-automatic mode (matches legacy behavior).
    """
    from torchrec.distributed.planner.types import ParameterConstraints

    es_cfg = config.distributed.embedding_sharding
    global_strategy = es_cfg.strategy
    default_spec = es_cfg.default_table_sharding
    stype_map = _stype_map()

    # Validate global_strategy upfront so we warn once, not per-table.
    if global_strategy != "auto" and stype_map.get(global_strategy) is None:
        logger.warning(
            f"Unknown global sharding strategy {global_strategy!r}; "
            f"falling back to auto."
        )
        global_strategy = "auto"

    constraints: dict[str, ParameterConstraints] = {}
    tables = config.model.resolved_embedding_tables()
    for t in tables:
        merged = _merge_sharding_specs(
            inline=t.sharding,
            default=default_spec,
            global_strategy=global_strategy,
        )
        if merged is None:
            continue  # fully auto for this table
        kwargs = _table_spec_to_constraint_kwargs(merged.spec, merged.sharding_type, stype_map)
        if not kwargs:
            continue
        constraints[t.name] = ParameterConstraints(**kwargs)

    return constraints if constraints else None


@dataclass
class _MergedShardingChoice:
    """Result of merging inline + default + global into one effective spec."""
    sharding_type: str | None
    spec: "object | None"   # TableShardingSpec or None when no other fields set


def _merge_sharding_specs(inline, default, global_strategy: str):
    """Field-by-field merge: inline overrides default; global supplies sharding_type only.

    Returns ``None`` when there is nothing to constrain (auto for this table).
    """
    # Resolve sharding_type via inline > default > global
    sharding_type = None
    if inline is not None and inline.sharding_type is not None:
        sharding_type = inline.sharding_type
    elif default is not None and default.sharding_type is not None:
        sharding_type = default.sharding_type
    elif global_strategy != "auto":
        sharding_type = global_strategy

    # If neither inline nor default has any non-sharding-type fields set, we
    # only need a sharding_type constraint (or nothing if even that is None).
    fielded_specs = [s for s in (inline, default) if s is not None]
    has_extra_fields = any(
        any(getattr(s, n) is not None for n in (
            "compute_kernel", "cache_load_factor", "min_partition",
            "enforce_hbm", "output_dtype", "ranks",
        ))
        for s in fielded_specs
    )
    if sharding_type is None and not has_extra_fields:
        return None

    # Merge non-sharding-type fields (inline wins per-field).
    if not has_extra_fields:
        merged_spec = None
    else:
        from primus_dlrm.config import TableShardingSpec
        def pick(name: str):
            if inline is not None and getattr(inline, name) is not None:
                return getattr(inline, name)
            if default is not None and getattr(default, name) is not None:
                return getattr(default, name)
            return None
        merged_spec = TableShardingSpec(
            sharding_type=sharding_type,
            compute_kernel=pick("compute_kernel"),
            cache_load_factor=pick("cache_load_factor"),
            min_partition=pick("min_partition"),
            enforce_hbm=pick("enforce_hbm"),
            output_dtype=pick("output_dtype"),
            ranks=pick("ranks"),
        )
    return _MergedShardingChoice(sharding_type=sharding_type, spec=merged_spec)


def _table_spec_to_constraint_kwargs(
    spec,
    sharding_type: str | None,
    stype_map: dict,
) -> dict:
    """Translate one ``TableShardingSpec`` (+ effective sharding_type) into
    the kwargs accepted by ``ParameterConstraints``."""
    from torchrec.distributed.planner.types import CacheParams
    try:
        from torchrec.distributed.embedding_types import DataType
    except ImportError:
        DataType = None

    kwargs: dict = {}
    if sharding_type is not None:
        stype_val = stype_map.get(sharding_type)
        if stype_val is None:
            logger.warning(f"Unknown sharding_type {sharding_type!r}; skipping.")
            return {}
        kwargs["sharding_types"] = [stype_val]

    if spec is None:
        return kwargs

    if spec.compute_kernel is not None:
        kernel = _COMPUTE_KERNEL_MAP.get(spec.compute_kernel, spec.compute_kernel)
        kwargs["compute_kernels"] = [kernel]

    if spec.cache_load_factor is not None:
        kwargs["cache_params"] = CacheParams(load_factor=spec.cache_load_factor)

    if spec.min_partition is not None:
        kwargs["min_partition"] = int(spec.min_partition)

    if spec.enforce_hbm is not None:
        kwargs["enforce_hbm"] = bool(spec.enforce_hbm)

    if spec.output_dtype is not None and DataType is not None:
        dtype_attr = _OUTPUT_DTYPE_MAP.get(spec.output_dtype.lower(), spec.output_dtype.upper())
        dtype_val = getattr(DataType, dtype_attr, None)
        if dtype_val is None:
            logger.warning(f"Unknown output_dtype {spec.output_dtype!r}; ignoring.")
        else:
            kwargs["output_dtype"] = dtype_val

    if spec.ranks:
        # TorchRec expects a single device_group string per table; pin via a
        # rank-derived group name. The planner consults this only for TW;
        # for other sharding types it is advisory.
        kwargs["device_group"] = f"rank{spec.ranks[0]}" if len(spec.ranks) == 1 \
            else "rank_" + "_".join(str(r) for r in spec.ranks)

    return kwargs


def _build_topology_kwargs(config, world_size: int) -> dict:
    """Resolve Topology kwargs with precedence env > YAML topology > torchrec default."""
    topo_cfg = getattr(config.distributed.embedding_sharding, "topology", None)
    yaml_hbm = topo_cfg.hbm_cap_gb if topo_cfg is not None else None
    yaml_ddr = topo_cfg.ddr_cap_gb if topo_cfg is not None else None
    yaml_lws = topo_cfg.local_world_size if topo_cfg is not None else None

    # Treat empty/whitespace env vars as unset — bash exporters frequently
    # forward ``FOO=""`` instead of unsetting the variable, and we want the
    # YAML topology to win in that case.
    def _env(name: str) -> str | None:
        v = os.environ.get(name)
        if v is None:
            return None
        v = v.strip()
        return v if v else None

    env_hbm = _env("PRIMUS_TORCHREC_HBM_CAP_GB")
    env_ddr = _env("PRIMUS_TORCHREC_DDR_CAP_GB")
    env_lws = _env("PRIMUS_TORCHREC_LOCAL_WORLD_SIZE")

    hbm = float(env_hbm) if env_hbm is not None else yaml_hbm
    ddr = float(env_ddr) if env_ddr is not None else yaml_ddr
    lws = int(env_lws) if env_lws is not None else yaml_lws

    kwargs: dict = {"world_size": world_size, "compute_device": "cuda"}
    if hbm is not None:
        kwargs["hbm_cap"] = int(float(hbm) * (1024 ** 3))
    if ddr is not None:
        kwargs["ddr_cap"] = int(float(ddr) * (1024 ** 3))
    if lws is not None:
        kwargs["local_world_size"] = int(lws)
    return kwargs


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
