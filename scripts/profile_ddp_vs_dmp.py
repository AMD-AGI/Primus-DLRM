#!/usr/bin/env python3
"""Profile per-phase step timing for DDP vs DMP.

Usage:
    torchrun --nproc_per_node=8 scripts/profile_ddp_vs_dmp.py \
        --config configs/dist_counter_v1.yaml --strategy ddp

    torchrun --nproc_per_node=8 scripts/profile_ddp_vs_dmp.py \
        --config configs/dist_counter_v1.yaml --strategy dmp --embedding-sharding auto
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import YambdaTrainDataset, collate_scoring_pairs
from primus_dlrm.distributed.setup import (
    cleanup, get_local_rank, get_rank, get_world_size,
    init_distributed, is_main_process, barrier,
)
from primus_dlrm.distributed.wrapper import wrap_model, is_dmp
from primus_dlrm.models.dlrm import DLRMBaseline
from primus_dlrm.training.losses import MultiTaskLoss


def build_model(config, num_users, num_items, num_artists, num_albums,
                audio_input_dim, device, tasks, meta_device=False):
    num_counter_windows = len(config.data.counter_windows_days) if config.data.enable_counters else 0
    kwargs = dict(
        config=config.model, num_users=num_users, num_items=num_items,
        num_artists=num_artists, num_albums=num_albums,
        audio_input_dim=audio_input_dim, device=device, tasks=tasks,
        num_counter_windows=num_counter_windows, meta_device=meta_device,
    )
    if config.model.model_type == "onetrans":
        from primus_dlrm.models.onetrans import OneTransModel
        return OneTransModel(**kwargs)
    return DLRMBaseline(**kwargs)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [rank %(process)d] %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


def _sync_event():
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", required=True, choices=["ddp", "dmp"])
    parser.add_argument("--embedding-sharding", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{get_local_rank()}")

    config = Config.load(args.config)
    torch.manual_seed(config.train.seed + rank)

    train_dataset = YambdaTrainDataset(config.data, Path("data/processed"))
    per_gpu_batch = config.train.batch_size // world_size
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(train_dataset, batch_size=per_gpu_batch, sampler=sampler,
                        num_workers=config.data.num_workers,
                        collate_fn=collate_scoring_pairs, pin_memory=True, drop_last=True)

    num_users = int(train_dataset.store.unique_uids.max()) + 1
    active_tasks = [k for k, v in config.train.loss_weights.items() if v > 0]
    use_dmp = args.strategy == "dmp"

    build_device = torch.device("cpu") if use_dmp else device
    model = build_model(config, num_users, train_dataset.num_items,
                        train_dataset.num_artists, train_dataset.num_albums,
                        train_dataset.audio_dim, build_device, active_tasks,
                        meta_device=use_dmp)

    model = wrap_model(model, device, dense_strategy=args.strategy,
                       embedding_sharding=args.embedding_sharding,
                       embedding_lr=config.train.embedding_lr,
                       embedding_weight_decay=config.train.weight_decay)

    if use_dmp:
        from torch.optim import AdamW
        dense_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(t in n for t in ("ebc.", "ec.", "embedding"))]
        optimizer = AdamW(dense_params, lr=config.train.lr, weight_decay=config.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr,
                                      weight_decay=config.train.weight_decay)

    loss_fn = MultiTaskLoss(weights=config.train.loss_weights)

    model.train()
    total_steps = args.warmup + args.steps
    step = 0

    times = {"h2d": [], "forward": [], "loss": [], "backward": [],
             "grad_clip": [], "optimizer": [], "total_gpu": []}

    barrier()
    if is_main_process():
        logger.info(f"Profiling {args.strategy} | {args.warmup} warmup + {args.steps} measured steps")

    for batch in loader:
        if step >= total_steps:
            break

        measuring = step >= args.warmup

        torch.cuda.synchronize()

        ev_start = _sync_event()

        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        ev_h2d = _sync_event()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(batch_gpu)
        ev_fwd = _sync_event()

        labels = {t: batch_gpu[t] for t in active_tasks}
        total_loss, _ = loss_fn(preds, labels)
        ev_loss = _sync_event()

        total_loss.backward()
        ev_bwd = _sync_event()

        nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            config.train.grad_clip,
        )
        ev_clip = _sync_event()

        optimizer.step()
        ev_opt = _sync_event()

        torch.cuda.synchronize()

        if measuring:
            times["h2d"].append(ev_start.elapsed_time(ev_h2d))
            times["forward"].append(ev_h2d.elapsed_time(ev_fwd))
            times["loss"].append(ev_fwd.elapsed_time(ev_loss))
            times["backward"].append(ev_loss.elapsed_time(ev_bwd))
            times["grad_clip"].append(ev_bwd.elapsed_time(ev_clip))
            times["optimizer"].append(ev_clip.elapsed_time(ev_opt))
            times["total_gpu"].append(ev_start.elapsed_time(ev_opt))

        step += 1

    if is_main_process():
        n_emb = sum(p.numel() for n, p in model.named_parameters()
                    if any(t in n for t in ("ebc.", "ec.", "embedding")))
        n_dense = sum(p.numel() for n, p in model.named_parameters()
                      if p.requires_grad and not any(t in n for t in ("ebc.", "ec.", "embedding")))
        n_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])

        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {args.strategy.upper()}"
                    f"{f' (embedding_sharding={args.embedding_sharding})' if use_dmp else ''}")
        logger.info(f"GPUs: {world_size}, per-GPU batch: {per_gpu_batch}")
        logger.info(f"Embedding params: {n_emb:,}, Dense params: {n_dense:,}")
        logger.info(f"Params in optimizer: {n_opt:,}")
        logger.info(f"Measured over {len(times['total_gpu'])} steps (after {args.warmup} warmup)")
        logger.info(f"{'='*60}")
        logger.info(f"{'Phase':<20} {'Mean (ms)':>10} {'Std (ms)':>10} {'% of step':>10}")
        logger.info(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")

        import statistics
        mean_total = statistics.mean(times["total_gpu"])
        for phase in ["h2d", "forward", "loss", "backward", "grad_clip", "optimizer", "total_gpu"]:
            vals = times[phase]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            pct = m / mean_total * 100
            logger.info(f"{phase:<20} {m:>10.2f} {s:>10.2f} {pct:>9.1f}%")

        throughput = per_gpu_batch * world_size / (mean_total / 1000)
        logger.info(f"\nThroughput: {throughput:,.0f} samples/s")
        logger.info(f"Mean step time: {mean_total:.2f} ms")

        grad_bytes = n_opt * 4
        logger.info(f"\nAllreduce volume (DDP gradient): {grad_bytes / 1e9:.3f} GB "
                    f"({n_opt:,} fp32 params)")

    cleanup()


if __name__ == "__main__":
    main()
