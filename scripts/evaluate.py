#!/usr/bin/env python3
"""Evaluate a trained DLRM++ model."""
import argparse
import logging
from pathlib import Path

import torch

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import YambdaEvalDataset
from primus_dlrm.evaluation.metrics import evaluate_ranking
from primus_dlrm.models.dlrm import DLRMBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DLRM++")
    parser.add_argument("--config", default="configs/dlrm_baseline.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="listen_plus", choices=["listen_plus", "like"])
    args = parser.parse_args()

    config = Config.load(args.config)
    device = torch.device(args.device)
    processed_dir = Path(args.processed_dir)

    logger.info("Loading eval dataset...")
    eval_dataset = YambdaEvalDataset(config.data, processed_dir)

    logger.info("Building model...")
    model = DLRMBaseline(
        config=config.model,
        num_users=len(eval_dataset.item_popularity),
        num_items=len(eval_dataset.item_popularity),
        num_artists=int(eval_dataset.item_to_artist.max()) + 1,
        num_albums=int(eval_dataset.item_to_album.max()) + 1,
        audio_input_dim=eval_dataset.audio_dim,
        device=device,
    )

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    logger.info(f"Evaluating on task={args.task}...")
    results = evaluate_ranking(model, eval_dataset, device, task=args.task)

    for k, v in sorted(results.items()):
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
