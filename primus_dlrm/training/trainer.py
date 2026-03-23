"""Training loop for DLRM++."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from primus_dlrm.config import Config
from primus_dlrm.models.base import BaseModel
from primus_dlrm.training.losses import MultiTaskLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Single-node training loop with BF16 AMP, LR scheduling, checkpointing."""

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        config: Config,
        eval_fn: callable | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.eval_fn = eval_fn

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        tc = config.train
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )

        total_steps = tc.epochs * len(train_loader)
        warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=tc.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - tc.warmup_steps, 1))
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[tc.warmup_steps])

        self.loss_fn = MultiTaskLoss(weights=tc.loss_weights)
        self.scaler = torch.amp.GradScaler(enabled=tc.bf16)
        self.use_amp = tc.bf16

        self.global_step = 0
        self.ckpt_dir = Path(tc.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "tb_logs"))

    def train(self) -> None:
        tc = self.config.train

        for epoch in range(tc.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            num_batches = 0

            for batch in self.train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss, task_losses = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1

                if self.global_step % tc.log_interval == 0:
                    self._log_step(loss, task_losses, epoch)
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"step={self.global_step} | loss={loss:.4f} | lr={lr:.6f} | "
                        + " | ".join(f"{k}={v:.4f}" for k, v in task_losses.items())
                    )

                self.global_step += 1

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(num_batches, 1)
            throughput = num_batches * tc.batch_size / epoch_time

            logger.info(
                f"Epoch {epoch}/{tc.epochs} | "
                f"loss={avg_loss:.4f} | "
                f"time={epoch_time:.1f}s | "
                f"throughput={throughput:.0f} samples/s"
            )
            self.writer.add_scalar("epoch/loss", avg_loss, epoch)
            self.writer.add_scalar("epoch/throughput", throughput, epoch)

            self._save_checkpoint(epoch)

            if self.eval_fn is not None and (epoch + 1) % tc.eval_interval == 0:
                metrics = self.eval_fn(self.model, self.device)
                for k, v in metrics.items():
                    self.writer.add_scalar(f"eval/{k}", v, epoch)
                logger.info(f"Eval metrics: {metrics}")

        self.writer.close()

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
        self.optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self.use_amp):
            predictions = self.model(batch)
            labels = {k: batch[k] for k in self.loss_fn.weights if k in batch}
            total_loss, task_losses = self.loss_fn(predictions, labels)

        self.scaler.scale(total_loss).backward()

        if self.config.train.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return total_loss.item(), {k: v.item() for k, v in task_losses.items()}

    def _log_step(self, loss: float, task_losses: dict[str, float], epoch: int) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/loss", loss, self.global_step)
        self.writer.add_scalar("train/lr", lr, self.global_step)
        for k, v in task_losses.items():
            self.writer.add_scalar(f"train/loss_{k}", v, self.global_step)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.ckpt_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint: {path}")
