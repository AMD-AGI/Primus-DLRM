#!/usr/bin/env python3
"""Distributed training entry point, compatible with torchrun.

Usage:
    torchrun --nproc_per_node=8 --nnodes=1 \\
        scripts/run_distributed.py --config configs/dist_onetrans_v6.yaml

    torchrun --nproc_per_node=8 --nnodes=2 --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        scripts/run_distributed.py --config configs/dist_onetrans_v6.yaml
"""
import argparse
import logging
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import (
    YambdaEvalDataset,
    YambdaTrainDataset,
    collate_to_dict,
)
from primus_dlrm.data.pipeline_batch import collate_pipeline_batch
from primus_dlrm.data.synthetic import SyntheticDataset, collate_synthetic
from primus_dlrm.distributed.setup import (
    barrier,
    cleanup,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)
from primus_dlrm.distributed.wrapper import wrap_model
from primus_dlrm.evaluation.metrics import evaluate_ranking, evaluate_ranking_peruser
from primus_dlrm.models.dlrm import DLRMBaseline
from primus_dlrm.schema import (
    FeatureSchema,
    build_schema_from_config,
    build_schema_from_synthetic,
)
from primus_dlrm.training.dist_trainer import DistributedTrainer
from primus_dlrm.training.runtime import configure_runtime


def build_model(config, schema: FeatureSchema, device=None, meta_device=False):
    """Build a model from config and schema.

    OneTrans is fully schema-driven.  DLRMBaseline still uses legacy params
    extracted from the schema's embedding tables.
    """
    if config.model.model_type == "onetrans":
        from primus_dlrm.models.onetrans import OneTransModel
        return OneTransModel(
            config=config.model, schema=schema,
            device=device, meta_device=meta_device,
        )
    return DLRMBaseline(
        config=config.model, schema=schema,
        device=device, meta_device=meta_device,
    )

_rank = os.environ.get("RANK", "?")
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s [rank {_rank}][pid %(process)d] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


class _InfiniteDataLoader:
    """Wraps a DataLoader to restart automatically when exhausted."""

    def __init__(self, loader):
        self.loader = loader
        self.sampler = loader.sampler

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        while True:
            yield from self.loader


def _setup_synthetic(config, world_size, rank, pipeline):
    """Build schema, dataset, and dataloader for synthetic data.

    Vocab sizes come from ``synthetic.embedding_tables`` in positional order,
    matching the schema tables defined in the YAML.
    """
    syn = config.data.synthetic
    vocab_sizes = [t.num_embeddings for t in syn.embedding_tables]
    schema = build_schema_from_config(config, vocab_sizes)

    if is_main_process():
        n_tables = len(schema.embedding_tables)
        n_feats = len(schema.scalar_features) + sum(
            len(f) for f in schema.sequence_groups.values()
        )
        n_dense = len(schema.dense_features)
        table_summary = ", ".join(
            f"{t.name}={t.num_embeddings}" for t in schema.embedding_tables
        )
        logger.info(
            f"Synthetic data: {n_tables} tables, {n_feats} sparse features, "
            f"{n_dense} dense features, {syn.num_samples} samples "
            f"({table_summary})"
        )

    dataset = SyntheticDataset(schema, syn)
    per_gpu_batch = config.train.batch_size // world_size
    # Disable shuffle for infinite synthetic data — random data doesn't
    # benefit from shuffling and DistributedSampler shuffling 10M indices
    # adds unnecessary overhead per epoch.
    shuffle = config.train.shuffle and not dataset._infinite
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=shuffle,
    )
    if pipeline:
        collate_fn = partial(collate_pipeline_batch, schema=schema)
    else:
        collate_fn = collate_synthetic

    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    if dataset._infinite:
        loader = _InfiniteDataLoader(loader)

    return schema, dataset, loader


def _setup_real_data(config, processed_dir, world_size, rank, pipeline):
    """Build schema, dataset, and dataloader for real Yambda data."""
    if is_main_process():
        logger.info("Loading training dataset...")
    dataset = YambdaTrainDataset(config.data, processed_dir)
    schema = build_schema_from_config(config, dataset.vocab_sizes)

    per_gpu_batch = config.train.batch_size // world_size
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=config.train.shuffle,
    )
    if pipeline:
        collate_fn = partial(collate_pipeline_batch, schema=schema)
    else:
        collate_fn = collate_to_dict
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return schema, dataset, loader


def main():
    parser = argparse.ArgumentParser(description="Distributed DLRM training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--dense-strategy", default="ddp", choices=["ddp", "fsdp", "dmp"])
    parser.add_argument("--embedding-sharding", default="auto",
                        choices=["auto", "table_wise", "row_wise", "data_parallel", "column_wise"],
                        help="Embedding sharding strategy for DMP mode")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--run-name", default="dist_run")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval after training (run eval separately)")
    parser.add_argument("--trace", action="store_true",
                        help="Dump per-step phase trace to <run>/trace.jsonl")
    parser.add_argument("--trace-steps", type=str, default="",
                        help="Comma-separated steps to capture traces (e.g. '500,1000'). "
                             "Each step gets its own trace file.")
    parser.add_argument("--pipeline", action="store_true",
                        help="Use TorchRec TrainPipelineSparseDist (3-stage, DMP only)")
    args = parser.parse_args()

    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}")

    config = Config.load(args.config)
    configure_runtime(config.train)
    torch.manual_seed(config.train.seed + rank)

    processed_dir = Path(args.processed_dir)
    run_dir = Path(args.results_dir) / args.run_name
    if is_main_process():
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)

    use_synthetic = config.data.synthetic.enabled
    use_dmp = args.dense_strategy == "dmp"
    # Multi-GPU DMP needs meta device so the planner can materialize shards.
    # Single-GPU DMP builds directly on GPU (no sharding needed).
    use_meta = use_dmp and world_size > 1
    build_device = torch.device("cpu") if use_meta else device

    if use_synthetic:
        schema, train_dataset, train_loader = _setup_synthetic(
            config, world_size, rank, args.pipeline,
        )
    else:
        schema, train_dataset, train_loader = _setup_real_data(
            config, processed_dir, world_size, rank, args.pipeline,
        )

    # Model
    if is_main_process():
        logger.info("Building model...")
    active_tasks = [k for k, v in config.train.loss_weights.items() if v > 0]

    model = build_model(
        config=config,
        schema=schema,
        device=build_device,
        meta_device=use_meta,
    )

    if use_dmp:
        num_params = sum(
            p.numel() for p in model.parameters() if not p.is_meta
        )
    else:
        num_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        strategy_label = args.dense_strategy
        if use_dmp:
            strategy_label = f"dmp(emb={args.embedding_sharding})"
        logger.info(f"Model: {num_params:,} params (dense), wrapping with {strategy_label} "
                    f"for {world_size} GPUs...")

    logger.info(f"Calling wrap_model with {args.dense_strategy} on rank {rank}")
    model = wrap_model(
        model, device,
        dense_strategy=args.dense_strategy,
        embedding_sharding=args.embedding_sharding,
        embedding_lr=config.train.embedding_lr,
        embedding_weight_decay=config.train.weight_decay,
        embedding_optimizer=config.train.embedding_optimizer,
        embedding_eps=config.train.embedding_eps,
    )
    logger.info(f"Model wrapped with {args.dense_strategy} on rank {rank}")

    # Eval function (rank 0 only, unless skipped; always skipped for synthetic)
    eval_fn = None
    if not args.skip_eval and not use_synthetic and is_main_process():
        eval_dataset = YambdaEvalDataset(config.data, processed_dir)
        pop = eval_dataset.item_popularity
        candidate_items = np.argsort(-pop)[:5000]

        eval_task = active_tasks[0] if active_tasks else "task0"

        def _eval(m, dev):
            rg = evaluate_ranking(
                m, eval_dataset, dev, candidate_item_ids=candidate_items,
                ks=[10, 50, 100], task=eval_task,
            )
            rp = evaluate_ranking_peruser(
                m, eval_dataset, dev, ks=[10, 50, 100],
                task=eval_task, top_n=100,
            )
            return {"global": rg, "peruser": rp}

        eval_fn = _eval

    barrier()

    # Train
    config.train.checkpoint_dir = str(run_dir)
    trainer = DistributedTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        eval_fn=eval_fn,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        trace=args.trace,
        trace_steps=[int(s) for s in args.trace_steps.split(",") if s.strip()] or None,
    )
    if args.pipeline and not use_dmp:
        raise ValueError("--pipeline requires --dense-strategy dmp")
    trainer.train(pipeline=args.pipeline)

    cleanup()
    if is_main_process():
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
