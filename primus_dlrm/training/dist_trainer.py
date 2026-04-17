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


def _get_system_metrics() -> str:
    """Collect GPU and CPU memory metrics for logging.

    GPU metrics use the same approach as Primus (MemoryStatsExtension):
      - HIP stats via torch.cuda.mem_get_info() (cheap, always available)
      - ROCm SMI VRAM stats via rocm-smi subprocess
    CPU metrics via psutil (optional).
    """
    GB = 1024 ** 3
    parts = []

    # HIP memory stats (same as Primus MemoryStatsExtension)
    try:
        hip_free, hip_total = torch.cuda.mem_get_info()
        hip_used = hip_total - hip_free
        hip_ratio = hip_used / hip_total * 100
        parts.append(
            f"hip mem usage/free/total/usage_ratio: "
            f"{hip_used / GB:.2f}GB/"
            f"{hip_free / GB:.2f}GB/"
            f"{hip_total / GB:.2f}GB/"
            f"{hip_ratio:.2f}%"
        )
    except Exception as e:
        logger.debug(f"HIP memory stats failed: {e}")

    # ROCm SMI VRAM stats + cross-rank max (same as Primus rocm_mem_info)
    try:
        import subprocess
        device_id = torch.cuda.current_device()
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", f"-d={device_id}"],
            text=True, timeout=5,
        )
        r_total, r_used = None, None
        for line in out.splitlines():
            if "Total Memory" in line:
                r_total = int(line.split(":")[-1].strip())
            elif "Total Used Memory" in line:
                r_used = int(line.split(":")[-1].strip())
        if r_total and r_used:
            r_free = r_total - r_used
            r_ratio = r_used / r_total * 100
            parts.append(
                f"rocm mem usage/free/total/usage_ratio: "
                f"{r_used / GB:.2f}GB/"
                f"{r_free / GB:.2f}GB/"
                f"{r_total / GB:.2f}GB/"
                f"{r_ratio:.2f}%"
            )
    except Exception as e:
        logger.debug(f"ROCm SMI stats failed: {e}")

    # CPU memory (psutil, optional)
    try:
        import psutil
        proc = psutil.Process()
        cpu_mem = proc.memory_info().rss / GB
        sys_mem = psutil.virtual_memory()
        parts.append(
            f"cpu_rss={cpu_mem:.2f}GB "
            f"sys_mem={sys_mem.used / GB:.2f}/{sys_mem.total / GB:.2f}GB"
        )
    except Exception as e:
        logger.debug(f"CPU memory stats failed: {e}")

    return " | ".join(parts) if parts else "metrics=N/A"


def _gather_max_gpu_memory() -> str:
    """All-gather GPU VRAM usage across ranks and return max usage string.

    Must be called on ALL ranks (not just rank 0) to avoid deadlock.
    Same approach as Primus MemoryStatsExtension.
    """
    try:
        if not torch.distributed.is_initialized():
            return ""
        hip_free, hip_total = torch.cuda.mem_get_info()
        hip_used = hip_total - hip_free
        usage_tensor = torch.tensor([hip_used], device="cuda", dtype=torch.float32)
        world_size = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(usage_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, usage_tensor)
        usages = [t.item() for t in gathered]
        max_usage = max(usages)
        max_rank = usages.index(max_usage)
        GB = 1024 ** 3
        return (
            f"max(rank-{max_rank}) GPU mem usage/usage_ratio: "
            f"{max_usage / GB:.2f}GB/"
            f"{max_usage / hip_total * 100:.2f}%"
        )
    except Exception as e:
        logger.debug(f"GPU memory all_gather failed: {e}")
        return ""


def _is_embedding_param(name: str) -> bool:
    """Heuristic to identify embedding parameters managed by DMP."""
    return any(tok in name for tok in ("ebc.", "ec.", "embedding"))


def _create_dense_optimizer(params: list, tc) -> torch.optim.Optimizer:
    """Create the dense parameter optimizer based on config.

    Supports:
      - "adamw": standard AdamW (default)
      - "shampoo": Distributed Shampoo (2nd-order, requires pip install distributed-shampoo)
    """
    if tc.dense_optimizer == "shampoo":
        from distributed_shampoo import DistributedShampoo
        from distributed_shampoo.shampoo_types import (
            AdaGradPreconditionerConfig,
            RootInvShampooPreconditionerConfig,
        )
        from distributed_shampoo.preconditioner.matrix_functions_types import EigenConfig

        preconditioner_config = (
            RootInvShampooPreconditionerConfig(
                amortized_computation_config=EigenConfig(),
                factor_matrix_dtype=torch.bfloat16,
                inverse_exponent_override={1: 0.5, 2: 0.5},
                num_tolerated_failed_amortized_computations=0,
            )
            if tc.shampoo_use_bf16_factor_matrix
            else RootInvShampooPreconditionerConfig()
        )

        logger.info(
            f"Using Distributed Shampoo (precondition_frequency={tc.shampoo_precondition_frequency}, "
            f"max_preconditioner_dim={tc.shampoo_max_preconditioner_dim}, "
            f"bf16_factor_matrix={tc.shampoo_use_bf16_factor_matrix}, "
            f"momentum={tc.shampoo_momentum})"
        )
        shampoo_kwargs: dict = dict(
            lr=tc.lr,
            betas=(0.9, 1.0),
            epsilon=1e-4,
            weight_decay=tc.weight_decay,
            max_preconditioner_dim=tc.shampoo_max_preconditioner_dim,
            precondition_frequency=tc.shampoo_precondition_frequency,
            start_preconditioning_step=tc.shampoo_precondition_frequency,
            grafting_config=AdaGradPreconditionerConfig(epsilon=1.4285714285714e-05),
            preconditioner_config=preconditioner_config,
        )
        # These kwargs were added in later Shampoo versions; only pass if non-default
        import inspect
        sig = inspect.signature(DistributedShampoo.__init__)
        for key, val in [
            ("momentum", tc.shampoo_momentum),
            ("use_nesterov", tc.shampoo_use_nesterov),
            ("use_bias_correction", True),
            ("use_decoupled_weight_decay", tc.shampoo_use_decoupled_weight_decay),
        ]:
            if key in sig.parameters:
                shampoo_kwargs[key] = val

        return DistributedShampoo(params, **shampoo_kwargs)

    elif tc.dense_optimizer == "adamw":
        return AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    
    raise ValueError(
        f"Unknown dense_optimizer='{tc.dense_optimizer}'. "
        f"Supported: 'adamw', 'shampoo'"
    )

    


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
        trace_steps: list[int] | None = None,
        trace_warmup: int = 5,
        trace_active: int = 10,
    ):
        self.model = model
        self._dataloader = train_loader
        self.config = config
        self.device = device
        self.eval_fn = eval_fn
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.tracer: Tracer | None = None
        if trace and is_main_process():
            trace_dir = Path(config.train.checkpoint_dir) / "trace"
            self.tracer = Tracer(
                trace_dir,
                trace_steps=trace_steps,
                warmup=trace_warmup,
                active=trace_active,
            )

        tc = config.train

        if is_dmp(model):
            dense_params = [
                p for n, p in model.named_parameters()
                if p.requires_grad and _is_embedding_param(n) is False
            ]
            self.optimizer = _create_dense_optimizer(dense_params, tc)
            self.fused_optimizer = model.fused_optimizer
            n_dense = sum(p.numel() for p in dense_params)
            n_emb = sum(
                p.numel() for n, p in model.named_parameters()
                if p.requires_grad and _is_embedding_param(n)
            )
            logger.info(f"DMP optimizer split: {n_dense:,} dense params ({tc.dense_optimizer}), "
                        f"{n_emb:,} embedding params (fused TBE)")
        else:
            self.optimizer = _create_dense_optimizer(
                list(self.model.parameters()), tc,
            )
            self.fused_optimizer = None

        # Sparse warmup (wraps the fused TBE optimizer if configured)
        if self.fused_optimizer is not None and tc.sparse_warmup_steps > 0:
            from torchrec.optim.warmup import WarmupOptimizer, WarmupStage
            self.fused_optimizer = WarmupOptimizer(
                self.fused_optimizer,
                [WarmupStage(max_iters=tc.sparse_warmup_steps, value=tc.sparse_warmup_value)],
                lr=tc.embedding_lr,
                param_name="__sparse_fused.__warmup",
            )
            logger.info(f"Sparse warmup: {tc.sparse_warmup_steps} steps, "
                        f"value={tc.sparse_warmup_value}")

        total_steps = tc.epochs * len(train_loader)
        warmup = LinearLR(self.optimizer, start_factor=tc.warmup_start_factor, total_iters=tc.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - tc.warmup_steps, 1))
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[tc.warmup_steps],
        )

        self.loss_fn = MultiTaskLoss(
            weights=tc.loss_weights,
            regression_tasks=tc.regression_tasks,
        )

        self.use_amp = tc.bf16
        # GradScaler improves BF16 gradient precision by scaling loss before
        # backward (preserving mantissa bits) then unscaling gradients before
        # the optimizer step.  Only used on the sequential path; the pipeline
        # path cannot use it because TorchRec's TrainPipelineSparseDist calls
        # backward() and optimizer.step() internally with no insertion point.
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

    # -- Step generators ------------------------------------------------------
    def _dataloader_stats(self) -> str:
        """Return dataloader queue depth and worker stats for monitoring.

        Only works with PyTorch's multi-process DataLoader (real data path).
        Returns empty string for synthetic data since it uses an
        _InfiniteDataLoader wrapper without internal queue tracking.
        """
        try:
            it = getattr(self._dataloader, '_iterator', None)
            if it is None:
                return ""
            rcvd = getattr(it, '_rcvd_idx', None)
            send = getattr(it, '_send_idx', None)
            if rcvd is not None and send is not None:
                queued = send - rcvd
                workers = getattr(it, '_num_workers', '?')
                return f"dl_queued={queued} dl_workers={workers}"
        except Exception as e:
            logger.warning(f"Failed to read dataloader stats: {e}")
        return ""

    # -- Forward + loss (shared by sequential and pipeline paths) -----------

    def _compute_loss(self, batch_dict, active_tasks):
        """Forward pass + loss computation.

        Shared by both sequential and pipeline training paths.
        Must be called inside torch.amp.autocast context.
        Returns (total_loss, task_losses) with total_loss as a GPU tensor.
        """
        if self.use_contrastive:
            inner = self.model.module if hasattr(self.model, "module") else self.model
            preds, cross_scores = inner.forward_with_cross_scores(
                batch_dict, cross_task=active_tasks[0],
            )
        else:
            preds = self.model(batch_dict)
            cross_scores = None

        labels = {t: batch_dict[t] for t in active_tasks}
        total_loss, task_losses = self.loss_fn(preds, labels)

        if cross_scores is not None:
            tc = self.config.train
            bpr_loss = self.contrastive_loss_fn(
                cross_scores, batch_dict[active_tasks[0]],
            )
            total_loss = total_loss + tc.contrastive_weight * bpr_loss
            task_losses["bpr"] = bpr_loss

        return total_loss, task_losses

    def _clip_gradients(self):
        """Clip dense parameter gradients by global norm."""
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        nn.utils.clip_grad_norm_(params, self.config.train.grad_clip)

    # -- Step generators ---------------------------------------------------
    # Each generator yields (loss_val: Tensor, task_losses: dict) per step.
    # loss_val is a detached GPU tensor; .item() is deferred to log intervals.

    def _sequential_steps(self, tc, active_tasks):
        """Manual forward/backward/optimizer loop over the dataloader."""
        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        for batch in self._dataloader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self.use_amp):
                total_loss, task_losses = self._compute_loss(batch, active_tasks)

            self.scaler.scale(total_loss).backward()

            if tc.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            yield total_loss.detach(), task_losses

    def _pipelined_steps(self, pipeline, dataloader_iter):
        """Yield steps from TorchRec TrainPipelineSparseDist.progress()."""
        while True:
            try:
                task_losses = pipeline.progress(dataloader_iter)
            except StopIteration:
                break

            if isinstance(task_losses, dict) and task_losses:
                loss_tensor = sum(
                    v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                    for v in task_losses.values()
                )
                loss_val = loss_tensor.detach()
            else:
                loss_val = torch.tensor(0.0, device=self.device)
            yield loss_val, task_losses if isinstance(task_losses, dict) else {}

    def _setup_pipeline(self, tc, active_tasks):
        """One-time setup for TorchRec 3-stage pipelined training.

        Returns the constructed TrainPipelineSparseDist object.
        """
        from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
        from torchrec.distributed.train_pipeline.tracing import (
            Tracer as _TorchRecTracer,
        )
        from primus_dlrm.training.pipeline_wrapper import PipelineModelWrapper

        # -- Monkey-patch TorchRec's FX tracer leaf-module detection ----------
        # TorchRec's tracer recognises "blocks" as a leaf module (don't trace
        # into it), but NOT its children like "blocks.0", "blocks.1", etc.
        # Without this patch, FX would try to symbolically trace into those
        # sub-modules and fail on transformer ops like dynamic shapes.
        _orig_is_leaf = _TorchRecTracer.is_leaf_module

        def _patched_is_leaf(self, m, module_qualified_name):
            for leaf in self._leaf_modules:
                if module_qualified_name.startswith(leaf + "."):
                    return True
            return _orig_is_leaf(self, m, module_qualified_name)

        _TorchRecTracer.is_leaf_module = _patched_is_leaf

        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        wrapped_model = PipelineModelWrapper(
            compute_loss_fn=lambda batch_dict: self._compute_loss(batch_dict, active_tasks),
            amp_dtype=amp_dtype,
            use_amp=self.use_amp,
        )

        # The pipeline calls backward+step atomically inside progress(), so we
        # register a full-backward hook to clip gradients right after autograd
        # finishes, before the optimizer step.
        if tc.grad_clip > 0:
            self.model.register_full_backward_hook(
                lambda _m, _gi, _go: self._clip_gradients()
            )

        # When _pipeline_mode=True the model reads embeddings from the
        # pre-built KJT instead of calling _build_unpooled_kjt at runtime,
        # giving FX a clean static graph to trace.
        inner = self.model.module if hasattr(self.model, "module") else self.model
        inner._pipeline_mode = True

        return TrainPipelineSparseDist(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            execute_all_batches=True,
            custom_model_fwd=wrapped_model,
        )

    # -- Unified training loop ------------------------------------------------

    def train(self, pipeline: bool = False) -> None:
        """Run training, either sequential or pipelined.

        Args:
            pipeline: If True, use TorchRec TrainPipelineSparseDist which
                overlaps H2D, embedding all-to-all, and dense fwd/bwd across
                3 CUDA streams.  Requires model wrapped with DMP.
        """
        tc = self.config.train
        active_tasks = [k for k, v in tc.loss_weights.items() if v > 0]
        per_gpu_batch = tc.batch_size // get_world_size()

        mode = "pipelined (TrainPipelineSparseDist)" if pipeline else "sequential"
        logger.info(f"Training mode: {mode}")

        pipeline_obj = None
        if pipeline:
            pipeline_obj = self._setup_pipeline(tc, active_tasks)

        for epoch in range(tc.epochs):
            sampler = self._dataloader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = torch.tensor(0.0, device=self.device)
            current_time = time.time()
            epoch_start = current_time
            window_start = current_time
            num_batches = 0

            if pipeline:
                steps = self._pipelined_steps(pipeline_obj, iter(self._dataloader))
            else:
                steps = self._sequential_steps(tc, active_tasks)

            for loss_val, task_losses in steps:
                self.scheduler.step()

                if self.tracer:
                    self.tracer.step()

                epoch_loss += loss_val.detach()
                num_batches += 1

                if self.global_step % self.log_interval == 0:
                    loss_scalar = loss_val.item()
                    max_mem_str = _gather_max_gpu_memory() if tc.log_max_gpu_memory else ""

                    if is_main_process():
                        current_time = time.time()
                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = current_time - epoch_start
                        throughput = num_batches * per_gpu_batch * get_world_size() / elapsed
                        window_elapsed = current_time - window_start
                        window_throughput = self.log_interval * per_gpu_batch * get_world_size() / window_elapsed if window_elapsed > 0 else 0
                        sys_metrics = _get_system_metrics()
                        dl_stats = self._dataloader_stats()
                        logger.info(
                            f"epoch={epoch} step={self.global_step} | "
                            f"loss={loss_scalar:.4f} | lr={lr:.6f} | "
                            f"global_throughput={throughput:.0f} samples/s | "
                            f"window_throughput={window_throughput:.0f} samples/s (over {self.log_interval} steps) | "
                            + " | ".join(
                                f"{k}={v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                                for k, v in task_losses.items()
                                if hasattr(v, "__float__")
                            )
                            + f" | {sys_metrics}"
                            + (f" | {max_mem_str}" if max_mem_str else "")
                            + (f" | {dl_stats}" if dl_stats else "")
                        )
                        window_start = current_time

                self.global_step += 1
                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    break

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss.item() / max(num_batches, 1)
            throughput = num_batches * per_gpu_batch * get_world_size() / epoch_time

            last_loss = loss_val.item()
            if is_main_process():
                logger.info(
                    f"=== Epoch {epoch} done === "
                    f"avg_loss={avg_loss:.4f} | last_loss={last_loss:.4f} | "
                    f"time={epoch_time:.1f}s | "
                    f"global_throughput={throughput:.0f} samples/s"
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
