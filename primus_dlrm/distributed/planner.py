"""Auto sharding planner for embedding tables and dense model strategy.

Recommends row-wise / table-wise / column-wise sharding per table,
and DDP vs FSDP for the dense model, based on model config + hardware topology.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from primus_dlrm.models.embedding import TableSpec

logger = logging.getLogger(__name__)

BYTES_PER_PARAM_FP32 = 4
OPTIMIZER_STATE_MULTIPLIER = 3  # Adam: param + m + v


@dataclass
class HardwareTopology:
    """Hardware specs for sharding decisions."""
    num_nodes: int = 1
    gpus_per_node: int = 8
    hbm_per_gpu_gb: float = 256.0
    intra_node_bw_gbps: float = 900.0  # xGMI / NVLink
    inter_node_bw_gbps: float = 100.0  # IB / RoCE

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node


@dataclass
class ShardingRecommendation:
    """Output of the planner for one embedding table."""
    table_name: str
    strategy: Literal["row_wise", "table_wise", "column_wise", "replicated"]
    reason: str
    table_bytes: int
    num_shards: int = 1


@dataclass
class DistributedPlan:
    """Complete distributed training plan."""
    embedding_plan: list[ShardingRecommendation] = field(default_factory=list)
    dense_strategy: Literal["ddp", "fsdp"] = "ddp"
    dense_reason: str = ""
    total_embedding_bytes: int = 0
    per_gpu_embedding_bytes: int = 0


def plan_sharding(
    table_specs: list[TableSpec],
    dense_param_count: int,
    topology: HardwareTopology | None = None,
) -> DistributedPlan:
    """Generate a sharding plan for embedding tables and dense model.

    Decision logic per table:
    - table_bytes > hbm_budget_per_gpu  -> row_wise  (split rows across GPUs)
    - embed_dim >= 256                  -> column_wise (split columns)
    - table_bytes < 10MB                -> table_wise  (assign whole table to one GPU)
    - else                              -> row_wise  (default for large tables)

    Dense strategy:
    - dense param memory (with optimizer state) < 10% of HBM -> DDP
    - else -> FSDP
    """
    if topology is None:
        topology = HardwareTopology()

    total_gpus = topology.total_gpus
    hbm_bytes = topology.hbm_per_gpu_gb * 1e9

    # Reserve 40% of HBM for activations + dense model + optimizer
    embedding_budget_per_gpu = hbm_bytes * 0.6

    plan = DistributedPlan()

    for spec in table_specs:
        table_bytes = spec.num_embeddings * spec.embedding_dim * BYTES_PER_PARAM_FP32
        table_bytes_with_opt = table_bytes * OPTIMIZER_STATE_MULTIPLIER

        if total_gpus == 1:
            rec = ShardingRecommendation(
                table_name=spec.name,
                strategy="replicated",
                reason="single GPU, no sharding needed",
                table_bytes=table_bytes,
                num_shards=1,
            )
        elif table_bytes_with_opt > embedding_budget_per_gpu:
            rec = ShardingRecommendation(
                table_name=spec.name,
                strategy="row_wise",
                reason=(
                    f"table too large for single GPU "
                    f"({table_bytes_with_opt / 1e9:.2f} GB with optimizer > "
                    f"{embedding_budget_per_gpu / 1e9:.1f} GB budget)"
                ),
                table_bytes=table_bytes,
                num_shards=total_gpus,
            )
        elif spec.embedding_dim >= 256:
            rec = ShardingRecommendation(
                table_name=spec.name,
                strategy="column_wise",
                reason=f"wide embedding (dim={spec.embedding_dim}), split columns",
                table_bytes=table_bytes,
                num_shards=min(total_gpus, spec.embedding_dim // 64),
            )
        elif table_bytes < 10 * 1024 * 1024:  # < 10 MB
            rec = ShardingRecommendation(
                table_name=spec.name,
                strategy="table_wise",
                reason=f"small table ({table_bytes / 1024:.0f} KB), assign to one GPU",
                table_bytes=table_bytes,
                num_shards=1,
            )
        else:
            rec = ShardingRecommendation(
                table_name=spec.name,
                strategy="row_wise",
                reason=f"default for medium/large table ({table_bytes / 1e6:.1f} MB)",
                table_bytes=table_bytes,
                num_shards=total_gpus,
            )

        plan.embedding_plan.append(rec)
        plan.total_embedding_bytes += table_bytes

    # Compute per-GPU embedding memory estimate
    if total_gpus > 1:
        per_gpu = sum(
            r.table_bytes / r.num_shards
            for r in plan.embedding_plan
        )
        plan.per_gpu_embedding_bytes = int(per_gpu)
    else:
        plan.per_gpu_embedding_bytes = plan.total_embedding_bytes

    # Dense strategy decision
    dense_bytes = dense_param_count * BYTES_PER_PARAM_FP32 * OPTIMIZER_STATE_MULTIPLIER
    dense_pct = dense_bytes / hbm_bytes * 100

    if dense_bytes > hbm_bytes * 0.1:
        plan.dense_strategy = "fsdp"
        plan.dense_reason = (
            f"dense model {dense_bytes / 1e9:.2f} GB "
            f"({dense_pct:.1f}% of HBM), using FSDP"
        )
    else:
        plan.dense_strategy = "ddp"
        plan.dense_reason = (
            f"dense model {dense_bytes / 1e6:.1f} MB "
            f"({dense_pct:.2f}% of HBM), DDP sufficient"
        )

    return plan


def log_plan(plan: DistributedPlan) -> None:
    """Print the sharding plan in a readable format."""
    logger.info("=" * 60)
    logger.info("SHARDING PLAN")
    logger.info("=" * 60)
    for rec in plan.embedding_plan:
        logger.info(
            f"  {rec.table_name:20s} | {rec.strategy:12s} | "
            f"shards={rec.num_shards:2d} | {rec.table_bytes / 1e6:8.1f} MB | "
            f"{rec.reason}"
        )
    logger.info(f"  Total embedding: {plan.total_embedding_bytes / 1e9:.2f} GB")
    logger.info(f"  Per-GPU embedding: {plan.per_gpu_embedding_bytes / 1e6:.1f} MB")
    logger.info(f"  Dense: {plan.dense_strategy} -- {plan.dense_reason}")
    logger.info("=" * 60)
