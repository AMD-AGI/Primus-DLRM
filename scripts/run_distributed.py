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
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import (
    YambdaEvalDataset,
    YambdaTrainDataset,
    collate_scoring_pairs,
)
from primus_dlrm.data.pipeline_batch import collate_pipeline_batch
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
from primus_dlrm.training.dist_trainer import DistributedTrainer


def build_model(config, num_users, num_items, num_artists, num_albums,
                audio_input_dim, device, tasks, meta_device=False):
    num_counter_windows = len(config.data.counter_windows_days) if config.data.enable_counters else 0
    kwargs = dict(
        config=config.model, num_users=num_users, num_items=num_items,
        num_artists=num_artists, num_albums=num_albums,
        audio_input_dim=audio_input_dim, device=device, tasks=tasks,
        num_counter_windows=num_counter_windows,
        meta_device=meta_device,
    )
    if config.model.model_type == "onetrans":
        from primus_dlrm.models.onetrans import OneTransModel
        return OneTransModel(**kwargs)
    return DLRMBaseline(**kwargs)

_rank = os.environ.get("RANK", "?")
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s [rank {_rank}][pid %(process)d] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--pipeline", action="store_true",
                        help="Use TorchRec TrainPipelineSparseDist (3-stage, DMP only)")
    args = parser.parse_args()

    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}")

    config = Config.load(args.config)
    torch.manual_seed(config.train.seed + rank)

    processed_dir = Path(args.processed_dir)
    run_dir = Path(args.results_dir) / args.run_name
    if is_main_process():
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)

    # Data
    if is_main_process():
        logger.info("Loading training dataset...")
    train_dataset = YambdaTrainDataset(config.data, processed_dir)

    per_gpu_batch = config.train.batch_size // world_size
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=config.train.shuffle,
    )
    # Pipeline needs DlrmBatch (Pipelineable) for CUDA stream management;
    # sequential uses plain dict from collate_scoring_pairs.
    collate_fn = collate_pipeline_batch if args.pipeline else collate_scoring_pairs
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    if is_main_process():
        logger.info("Building model...")
    num_users = int(train_dataset.store.unique_uids.max()) + 1
    active_tasks = [k for k, v in config.train.loss_weights.items() if v > 0]
    use_dmp = args.dense_strategy == "dmp"

    # DMP requires embedding modules on meta device so it can materialize
    # weights on the correct GPU per the sharding plan.
    build_device = torch.device("cpu") if use_dmp else device
    model = build_model(
        config=config,
        num_users=num_users,
        num_items=train_dataset.num_items,
        num_artists=train_dataset.num_artists,
        num_albums=train_dataset.num_albums,
        audio_input_dim=train_dataset.audio_dim,
        device=build_device,
        tasks=active_tasks,
        meta_device=use_dmp,
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

    model = wrap_model(
        model, device,
        dense_strategy=args.dense_strategy,
        embedding_sharding=args.embedding_sharding,
        embedding_lr=config.train.embedding_lr,
        embedding_weight_decay=config.train.weight_decay,
    )

    # Eval function (rank 0 only, unless skipped)
    eval_fn = None
    if not args.skip_eval and is_main_process():
        eval_dataset = YambdaEvalDataset(config.data, processed_dir)
        pop = eval_dataset.item_popularity
        candidate_items = np.argsort(-pop)[:5000]

        def _eval(m, dev):
            rg = evaluate_ranking(
                m, eval_dataset, dev, candidate_item_ids=candidate_items,
                ks=[10, 50, 100], task="listen_plus",
            )
            rp = evaluate_ranking_peruser(
                m, eval_dataset, dev, ks=[10, 50, 100],
                task="listen_plus", top_n=100,
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
    )
    if args.pipeline:
        if not use_dmp:
            raise ValueError("--pipeline requires --dense-strategy dmp")
        trainer.train_pipelined()
    else:
        trainer.train()

    cleanup()
    if is_main_process():
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
