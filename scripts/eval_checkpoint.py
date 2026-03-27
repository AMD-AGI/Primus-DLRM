#!/usr/bin/env python3
"""Evaluate a checkpoint on a single GPU.

Works for both DDP and FSDP checkpoints (FSDP checkpoints are saved as
full consolidated state dicts, so they load into a plain model).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoint.py \
        --config configs/dist_onetrans_v6.yaml \
        --checkpoint results/s2_onetrans_8gpu_ddp/checkpoints/epoch_0.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import YambdaEvalDataset, YambdaTrainDataset
from primus_dlrm.evaluation.metrics import evaluate_ranking, evaluate_ranking_peruser
from primus_dlrm.models.dlrm import DLRMBaseline
from primus_dlrm.models.onetrans import OneTransModel
from primus_dlrm.schema import build_schema_from_config

# force=True required: imported modules configure the root logger first,
# making a second basicConfig() a no-op without it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=5000,
                        help="Number of candidate items for ranking eval (lower = faster)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log progress every N users during eval")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    config = Config.load(args.config)
    processed_dir = Path(args.processed_dir)

    train_dataset = YambdaTrainDataset(config.data, processed_dir)
    num_users = int(train_dataset.store.unique_uids.max()) + 1
    active_tasks = [k for k, v in config.train.loss_weights.items() if v > 0]
    schema = build_schema_from_config(config, {
        "item": train_dataset.num_items, "artist": train_dataset.num_artists,
        "album": train_dataset.num_albums, "uid": num_users,
    })

    if config.model.model_type == "onetrans":
        model = OneTransModel(config=config.model, schema=schema, device=device)
    else:
        model = DLRMBaseline(
            config=config.model, num_users=num_users,
            num_items=train_dataset.num_items,
            num_artists=train_dataset.num_artists,
            num_albums=train_dataset.num_albums,
            audio_input_dim=train_dataset.audio_dim,
            device=device, tasks=active_tasks,
            num_counter_windows=num_counter_windows,
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: step={ckpt['global_step']}, epoch={ckpt['epoch']}")

    import time

    eval_dataset = YambdaEvalDataset(config.data, processed_dir)
    pop = eval_dataset.item_popularity
    candidate_items = np.argsort(-pop)[:args.num_candidates]
    logger.info(
        f"Eval config: {len(eval_dataset)} users, "
        f"{len(candidate_items)} candidates, ks=[10, 50, 100]"
    )

    logger.info("Starting global ranking eval...")
    t0 = time.time()
    with torch.no_grad():
        rg = evaluate_ranking(
            model, eval_dataset, device,
            candidate_item_ids=candidate_items,
            ks=[10, 50, 100], task="listen_plus",
            log_interval=args.log_interval,
        )
    logger.info(f"Global eval done in {time.time() - t0:.1f}s")

    logger.info("Starting per-user ranking eval...")
    t0 = time.time()
    with torch.no_grad():
        rp = evaluate_ranking_peruser(
            model, eval_dataset, device,
            ks=[10, 50, 100], task="listen_plus", top_n=100,
            log_interval=args.log_interval,
        )
    logger.info(f"Per-user eval done in {time.time() - t0:.1f}s")

    label = f"global-{args.num_candidates}"
    logger.info(f"[{label}] " + " | ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(rg.items())
    ))
    logger.info("[peruser-100] " + " | ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(rp.items())
    ))


if __name__ == "__main__":
    main()
