#!/usr/bin/env python3
"""Train DLRM++ or OneTrans on Yambda."""
import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import YambdaTrainDataset, YambdaEvalDataset, collate_scoring_pairs
from primus_dlrm.evaluation.metrics import evaluate_ranking
from primus_dlrm.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def build_model(config, num_users, num_items, num_artists, num_albums, audio_input_dim, device):
    if config.model.model_type == "onetrans":
        from primus_dlrm.models.onetrans import OneTransModel
        return OneTransModel(
            config=config.model, num_users=num_users, num_items=num_items,
            num_artists=num_artists, num_albums=num_albums,
            audio_input_dim=audio_input_dim, device=device,
        )
    else:
        from primus_dlrm.models.dlrm import DLRMBaseline
        return DLRMBaseline(
            config=config.model, num_users=num_users, num_items=num_items,
            num_artists=num_artists, num_albums=num_albums,
            audio_input_dim=audio_input_dim, device=device,
        )


def main():
    parser = argparse.ArgumentParser(description="Train DLRM++ / OneTrans")
    parser.add_argument("--config", default="configs/dlrm_baseline.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = Config.load(args.config)
    device = torch.device(args.device)
    processed_dir = Path(args.processed_dir)

    torch.manual_seed(config.train.seed)

    logger.info("Loading training dataset...")
    train_dataset = YambdaTrainDataset(config.data, processed_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_scoring_pairs,
        pin_memory=True,
        drop_last=True,
    )

    logger.info("Loading eval dataset...")
    eval_dataset = YambdaEvalDataset(config.data, processed_dir)

    logger.info(f"Building model (type={config.model.model_type})...")
    num_users = int(train_dataset.store.unique_uids.max()) + 1
    model = build_model(
        config, num_users=num_users,
        num_items=train_dataset.num_items,
        num_artists=train_dataset.num_artists,
        num_albums=train_dataset.num_albums,
        audio_input_dim=train_dataset.audio_dim,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    def eval_fn(model, device):
        return evaluate_ranking(model, eval_dataset, device)

    trainer = Trainer(model, train_loader, config, eval_fn=eval_fn, device=device)

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
