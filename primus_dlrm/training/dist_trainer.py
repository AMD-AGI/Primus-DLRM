"""Distributed training loop for DLRM/OneTrans with DDP/FSDP/DMP.

Supports two training modes:
  - Sequential (default): standard for-loop with manual fwd/bwd/optim.
  - Pipelined (--pipeline): TorchRec TrainPipelineSparseDist overlapping
    H2D transfer, embedding all-to-all, and dense fwd/bwd on 3 CUDA streams.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from primus_dlrm.config import Config
from primus_dlrm.distributed.setup import barrier, get_rank, get_world_size, is_main_process
from primus_dlrm.distributed.wrapper import is_dmp
from primus_dlrm.training.losses import InBatchBPRLoss, MultiTaskLoss
from primus_dlrm.training.tracer import Tracer

logger = logging.getLogger(__name__)


def _unwrap(model: nn.Module) -> nn.Module:
    """Get the underlying model from DDP / FSDP / DMP."""
    if isinstance(model, FSDP):
        return model
    return model.module if hasattr(model, "module") else model


def _collect_dmp_state_dict(model: nn.Module) -> dict:
    """Collect DMP state dict, gathering ShardedTensors into regular tensors on rank 0."""
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from primus_dlrm.distributed.setup import get_local_rank

    # Use local_rank (not global rank) for device -- on node 2, rank 8
    # maps to cuda:0, not cuda:8 which doesn't exist locally.
    device = torch.device(f"cuda:{get_local_rank()}")
    raw_sd = _unwrap(model).state_dict()
    collected = {}
    for key, val in raw_sd.items():
        if isinstance(val, ShardedTensor):
            full = torch.zeros(val.metadata().size, dtype=val.dtype, device=device)
            for shard in val.local_shards():
                md = shard.metadata
                slices = tuple(
                    slice(off, off + size)
                    for off, size in zip(md.shard_offsets, md.shard_sizes)
                )
                full[slices] = shard.tensor.to(device)
            torch.distributed.reduce(full, dst=0, op=torch.distributed.ReduceOp.SUM)
            collected[key] = full.cpu()
        else:
            collected[key] = val.cpu() if isinstance(val, torch.Tensor) else val
    return collected


def _is_fsdp(model: nn.Module) -> bool:
    return isinstance(model, FSDP)


def _is_embedding_param(name: str) -> bool:
    """Heuristic to identify embedding parameters managed by DMP."""
    return any(tok in name for tok in ("ebc.", "ec.", "embedding"))


class DistributedTrainer:
    """Multi-GPU training loop with DistributedSampler + rank-aware logging.

    The model should already be wrapped with DDP/FSDP/DMP before passing here.
    Global batch size = per-GPU batch_size * world_size.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: Config,
        device: torch.device,
        eval_fn: callable | None = None,
        max_steps: int = 0,
        log_interval: int = 1000,
        trace: bool = False,
        trace_warmup: int = 5,
        trace_active: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.eval_fn = eval_fn
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.tracer: Tracer | None = None
        if trace and is_main_process():
            trace_dir = Path(config.train.checkpoint_dir) / "trace"
            self.tracer = Tracer(trace_dir, warmup=trace_warmup, active=trace_active)

        tc = config.train

        if is_dmp(model):
            dense_params = [
                p for n, p in model.named_parameters()
                if p.requires_grad and _is_embedding_param(n) is False
            ]
            self.optimizer = AdamW(
                dense_params, lr=tc.lr, weight_decay=tc.weight_decay,
            )
            self.fused_optimizer = model.fused_optimizer
            n_dense = sum(p.numel() for p in dense_params)
            n_emb = sum(
                p.numel() for n, p in model.named_parameters()
                if p.requires_grad and _is_embedding_param(n)
            )
            logger.info(f"DMP optimizer split: {n_dense:,} dense params (AdamW), "
                        f"{n_emb:,} embedding params (fused TBE)")
        else:
            self.optimizer = AdamW(
                self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay,
            )
            self.fused_optimizer = None

        total_steps = tc.epochs * len(train_loader)
        warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=tc.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - tc.warmup_steps, 1))
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[tc.warmup_steps],
        )

        self.loss_fn = MultiTaskLoss(weights=tc.loss_weights)

        self.use_amp = tc.bf16
        self.scaler = torch.amp.GradScaler(enabled=tc.bf16)

        self.use_contrastive = tc.contrastive_weight > 0
        self.contrastive_loss_fn: InBatchBPRLoss | None = None
        if self.use_contrastive:
            self.contrastive_loss_fn = InBatchBPRLoss(
                temperature=tc.contrastive_temperature,
            )

        self.global_step = 0
        self.ckpt_dir = Path(tc.checkpoint_dir) / "checkpoints"
        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        tc = self.config.train
        active_tasks = [k for k, v in tc.loss_weights.items() if v > 0]
        per_gpu_batch = tc.batch_size // get_world_size()

        for epoch in range(tc.epochs):
            sampler = self.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            num_batches = 0

            for batch in self.train_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                self.optimizer.zero_grad()
                amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self.use_amp):
                    if self.use_contrastive:
                        inner = self.model.module if hasattr(self.model, "module") else self.model
                        preds, cross_scores = inner.forward_with_cross_scores(
                            batch, cross_task="listen_plus",
                        )
                    else:
                        preds = self.model(batch)
                    labels = {t: batch[t] for t in active_tasks}
                    total_loss, task_losses = self.loss_fn(preds, labels)

                    if self.use_contrastive:
                        bpr_loss = self.contrastive_loss_fn(
                            cross_scores, batch["listen_plus"],
                        )
                        total_loss = total_loss + tc.contrastive_weight * bpr_loss
                        task_losses["bpr"] = bpr_loss

                self.scaler.scale(total_loss).backward()

                if tc.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    params_to_clip = (
                        [p for g in self.optimizer.param_groups for p in g["params"]]
                    )
                    nn.utils.clip_grad_norm_(params_to_clip, tc.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                if self.tracer:
                    self.tracer.step()

                loss_val = total_loss.item()
                epoch_loss += loss_val
                num_batches += 1

                if self.global_step % self.log_interval == 0 and is_main_process():
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - epoch_start
                    throughput = num_batches * per_gpu_batch * get_world_size() / elapsed
                    logger.info(
                        f"epoch={epoch} step={self.global_step} | "
                        f"loss={loss_val:.4f} | lr={lr:.6f} | "
                        f"throughput={throughput:.0f} samples/s | "
                        + " | ".join(
                            f"{k}={v:.4f}"
                            for k, v in task_losses.items()
                            if hasattr(v, "__float__")
                        )
                    )

                self.global_step += 1

                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    break

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(num_batches, 1)
            throughput = num_batches * per_gpu_batch * get_world_size() / epoch_time

            if is_main_process():
                logger.info(
                    f"=== Epoch {epoch} done === "
                    f"avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s | "
                    f"throughput={throughput:.0f} samples/s"
                )

            barrier()

            self._save_checkpoint(epoch)

            if self.eval_fn is not None:
                self._run_eval()

            barrier()

            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        if self.tracer:
            self.tracer.stop()
            files = self.tracer.trace_files
            logger.info(f"Trace dumped: {self.tracer.trace_dir}/ ({len(files)} files)")

    def train_pipelined(self) -> None:
        """3-stage pipelined training using TorchRec TrainPipelineSparseDist.

        Overlaps H2D transfer (memcpy stream), sparse data distribution /
        embedding all-to-all (data_dist stream), and dense forward+backward
        (default stream) across consecutive batches.

        Requires: model wrapped with DMP (DistributedModelParallel).
        """
        from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
        from torchrec.distributed.train_pipeline.tracing import (
            Tracer as _TorchRecTracer,
        )
        from primus_dlrm.data.pipeline_batch import DlrmBatch
        from primus_dlrm.training.pipeline_wrapper import PipelineModelWrapper

        # -- Monkey-patch TorchRec's FX tracer leaf-module detection ----------
        # TorchRec's tracer recognises "blocks" as a leaf module (don't trace
        # into it), but NOT its children like "blocks.0", "blocks.1", etc.
        # Without this patch, FX would try to symbolically trace into those
        # sub-modules and fail on transformer ops like dynamic shapes.
        _orig_is_leaf = _TorchRecTracer.is_leaf_module

        def _patched_is_leaf(self, m, module_qualified_name):
            # If a parent qualified name is already a leaf, treat all
            # descendants (children, grandchildren, …) as leaves too.
            for leaf in self._leaf_modules:
                if module_qualified_name.startswith(leaf + "."):
                    return True
            return _orig_is_leaf(self, m, module_qualified_name)

        _TorchRecTracer.is_leaf_module = _patched_is_leaf

        tc = self.config.train
        active_tasks = [k for k, v in tc.loss_weights.items() if v > 0]
        per_gpu_batch = tc.batch_size // get_world_size()

        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        # -- Wrap model + loss into a single callable for the pipeline --------
        # TrainPipelineSparseDist invokes `custom_model_fwd(batch)` on the
        # default CUDA stream after embeddings are ready.  PipelineModelWrapper
        # bundles forward + loss computation so the pipeline gets back the loss
        # (needed for backward) and per-task loss dict (for logging).
        wrapped_model = PipelineModelWrapper(
            model=self.model,
            loss_fn=self.loss_fn,
            active_tasks=active_tasks,
            contrastive_loss_fn=self.contrastive_loss_fn,
            contrastive_weight=tc.contrastive_weight,
            amp_dtype=amp_dtype,
            use_amp=self.use_amp,
        )

        # -- Gradient clipping via backward hook ------------------------------
        # In non-pipelined training we clip between backward() and step().
        # The pipeline calls backward+step atomically inside progress(), so we
        # cannot inject code between them.  A full-backward hook fires right
        # after autograd finishes, before the optimizer step, giving us the
        # same clipping semantics.
        if tc.grad_clip > 0:
            _clip_value = tc.grad_clip
            _optimizer = self.optimizer

            def _post_backward_grad_clip(*_args):
                params = [p for g in _optimizer.param_groups for p in g["params"]]
                nn.utils.clip_grad_norm_(params, _clip_value)

            self.model.register_full_backward_hook(
                lambda _m, _gi, _go: _post_backward_grad_clip()
            )

        # -- Toggle the model's embedding path for FX tracing ----------------
        # When _pipeline_mode=True the model's forward reads embeddings from
        # the pre-built KJT (batch.unpooled_kjt → EmbeddingCollection) instead
        # of calling _build_unpooled_kjt at runtime.  This gives FX a clean
        # static graph it can trace: DlrmBatch["unpooled_kjt"] → ec(kjt).
        inner = self.model.module if hasattr(self.model, "module") else self.model
        inner._pipeline_mode = True

        # -- Construct the 3-stage pipeline -----------------------------------
        # Stage 0 (memcpy stream):    H2D transfer of next batch
        # Stage 1 (data_dist stream): KJT redistribution + embedding all-to-all
        # Stage 2 (default stream):   dense fwd/bwd/optim on current batch
        #
        # execute_all_batches=True ensures the pipeline drains completely at
        # epoch end instead of dropping the last 2 in-flight batches.
        # custom_model_fwd overrides the default `model(batch)` call with our
        # wrapper that also computes the loss.
        pipeline = TrainPipelineSparseDist(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            execute_all_batches=True,
            custom_model_fwd=wrapped_model,
        )

        for epoch in range(tc.epochs):
            sampler = self.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                # Each epoch must reshuffle with a different seed so ranks
                # see different data orderings across epochs.
                sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            num_batches = 0

            # Fresh iterator each epoch; the pipeline pulls batches lazily via
            # next(dataloader_iter) inside progress().
            dataloader_iter = iter(self.train_loader)
            while True:
                try:
                    # progress() advances the 3-stage pipeline by one step:
                    #   - Kicks off H2D for batch N+2
                    #   - Kicks off data_dist (all-to-all) for batch N+1
                    #   - Runs fwd/bwd/optim for batch N on the default stream
                    # Returns the output of custom_model_fwd for batch N,
                    # which is the per-task loss dict from PipelineModelWrapper.
                    task_losses = pipeline.progress(dataloader_iter)
                except StopIteration:
                    break

                # The pipeline handles optimizer.step() internally, but the LR
                # scheduler is our responsibility — step it after each batch.
                self.scheduler.step()

                if self.tracer:
                    self.tracer.step()

                num_batches += 1

                # task_losses is a dict like {"listen_plus": 0.42, "like": 0.31, ...}.
                # During pipeline warmup (first 2 calls) it may be empty/None
                # because no batch has completed the full 3-stage journey yet.
                if isinstance(task_losses, dict) and task_losses:
                    loss_val = sum(
                        v.item() if isinstance(v, torch.Tensor) else v
                        for v in task_losses.values()
                    )
                else:
                    loss_val = 0.0
                epoch_loss += loss_val

                if self.global_step % self.log_interval == 0 and is_main_process():
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - epoch_start
                    throughput = num_batches * per_gpu_batch * get_world_size() / elapsed
                    logger.info(
                        f"epoch={epoch} step={self.global_step} | "
                        f"loss={loss_val:.4f} | lr={lr:.6f} | "
                        f"throughput={throughput:.0f} samples/s | "
                        + " | ".join(
                            f"{k}={v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                            for k, v in task_losses.items()
                            if hasattr(v, "__float__")
                        )
                    )

                self.global_step += 1
                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    break

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(num_batches, 1)
            throughput = num_batches * per_gpu_batch * get_world_size() / epoch_time

            if is_main_process():
                logger.info(
                    f"=== Epoch {epoch} done === "
                    f"avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s | "
                    f"throughput={throughput:.0f} samples/s"
                )

            # All ranks must reach the barrier before checkpointing so that
            # the gathered state dict is consistent across shards.
            barrier()
            self._save_checkpoint(epoch)

            if self.eval_fn is not None:
                self._run_eval()

            # Second barrier ensures eval/checkpoint I/O is finished on all
            # ranks before any rank begins the next epoch's data loading.
            barrier()
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        if self.tracer:
            self.tracer.stop()
            files = self.tracer.trace_files
            logger.info(f"Trace dumped: {self.tracer.trace_dir}/ ({len(files)} files)")

    def _run_eval(self) -> None:
        """Run evaluation. For FSDP, uses summon_full_params so all ranks
        participate in the all-gather, then only rank 0 runs eval."""
        if _is_fsdp(self.model):
            with FSDP.summon_full_params(self.model, writeback=False, rank0_only=True):
                if is_main_process():
                    self.model.eval()
                    with torch.no_grad():
                        metrics = self.eval_fn(self.model, self.device)
                    self._log_eval_metrics(metrics)
        elif is_main_process():
            eval_model = _unwrap(self.model)
            eval_model.eval()
            with torch.no_grad():
                metrics = self.eval_fn(eval_model, self.device)
            self._log_eval_metrics(metrics)

    def _log_eval_metrics(self, metrics) -> None:
        if isinstance(metrics, dict) and "global" in metrics:
            gl = metrics["global"]
            pu = metrics["peruser"]
            logger.info(
                "  [global-5000] " + " | ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(gl.items())
                )
            )
            logger.info(
                "  [peruser-100] " + " | ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(pu.items())
                )
            )
        else:
            logger.info(f"Eval metrics: {metrics}")

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.ckpt_dir / f"epoch_{epoch}.pt"
        if _is_fsdp(self.model):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state = self.model.state_dict()
                if is_main_process():
                    torch.save(
                        {"epoch": epoch, "global_step": self.global_step,
                         "model_state_dict": state},
                        path,
                    )
                    logger.info(f"Saved FSDP full checkpoint: {path}")
        elif is_dmp(self.model):
            state = _collect_dmp_state_dict(self.model)
            if is_main_process():
                torch.save(
                    {"epoch": epoch, "global_step": self.global_step,
                     "model_state_dict": state},
                    path,
                )
                logger.info(f"Saved DMP checkpoint: {path}")
        elif is_main_process():
            state = _unwrap(self.model).state_dict()
            torch.save(
                {"epoch": epoch, "global_step": self.global_step,
                 "model_state_dict": state},
                path,
            )
            logger.info(f"Saved checkpoint: {path}")
