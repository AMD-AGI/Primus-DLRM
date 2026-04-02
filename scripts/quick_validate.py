#!/usr/bin/env python3
"""Quick validation: train for N steps, save checkpoint, run eval."""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import YambdaTrainDataset, YambdaEvalDataset, collate_to_dict
from primus_dlrm.evaluation.metrics import evaluate_ranking
from primus_dlrm.models.dlrm import DLRMBaseline
from primus_dlrm.training.losses import MultiTaskLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrm_quick.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--eval-top-k", type=int, default=5000,
                        help="Number of popular items to score against for eval")
    args = parser.parse_args()

    config = Config.load(args.config)
    device = torch.device(args.device)
    processed_dir = Path(args.processed_dir)
    torch.manual_seed(config.train.seed)

    logger.info("Loading training dataset...")
    train_dataset = YambdaTrainDataset(config.data, processed_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True,
        num_workers=config.data.num_workers, collate_fn=collate_to_dict,
        pin_memory=True, drop_last=True,
    )

    logger.info("Building model...")
    if config.model.model_type == "onetrans":
        from primus_dlrm.models.onetrans import OneTransModel
        model = OneTransModel(config=config, device=device)
    else:
        model = DLRMBaseline(config=config, device=device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    loss_fn = MultiTaskLoss(
        weights=config.train.loss_weights,
        regression_tasks=config.train.regression_tasks,
    )

    # --- Train ---
    logger.info(f"Training for {args.max_steps} steps...")
    model.train()
    step = 0
    t0 = time.time()

    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(batch)
            labels = {k: batch[k] for k in loss_fn.weights if k in batch}
            total_loss, task_losses = loss_fn(preds, labels)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            throughput = (step + 1) * config.train.batch_size / elapsed if elapsed > 0 else 0
            logger.info(
                f"step={step} | loss={total_loss.item():.4f} | "
                f"throughput={throughput:.0f} samples/s | "
                + " | ".join(f"{k}={v:.4f}" for k, v in task_losses.items())
            )

        step += 1
        if step >= args.max_steps:
            break

    elapsed = time.time() - t0
    logger.info(f"Training done: {step} steps in {elapsed:.1f}s ({step * config.train.batch_size / elapsed:.0f} samples/s)")

    # --- Save ---
    ckpt_path = Path("checkpoints/quick_validate.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "step": step}, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # --- Eval ---
    logger.info("Loading eval dataset...")
    eval_dataset = YambdaEvalDataset(config.data, processed_dir)
    logger.info(f"Eval dataset: {len(eval_dataset)} test users")

    import numpy as np
    pop = eval_dataset.item_popularity
    candidate_items = np.argsort(-pop)[:args.eval_top_k]
    logger.info(f"Scoring against top-{args.eval_top_k} popular items")

    logger.info("Running evaluation...")
    results = evaluate_ranking(
        model, eval_dataset, device,
        candidate_item_ids=candidate_items,
        ks=[10, 50, 100],
        task="listen_plus",
    )
    logger.info("=== Evaluation Results (listen_plus) ===")
    for k, v in sorted(results.items()):
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_like = evaluate_ranking(
        model, eval_dataset, device,
        candidate_item_ids=candidate_items,
        ks=[10, 50, 100],
        task="like",
    )
    logger.info("=== Evaluation Results (like) ===")
    for k, v in sorted(results_like.items()):
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
