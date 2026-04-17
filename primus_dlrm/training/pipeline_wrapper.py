"""Runtime model+loss wrapper for TrainPipelineSparseDist.

Thin shim that wraps AMP autocast around the trainer's ``_compute_loss``
method.  All forward+loss+contrastive logic lives in
``DistributedTrainer._compute_loss`` — this module only adapts it for
TorchRec's pipeline interface.

The pipeline calls ``custom_model_fwd(batch)`` on the default stream during
Stage 2 of each step.  It expects the return value's first element to be the
scalar loss (used by ``backward()``); any additional elements are forwarded as
the return value of ``pipeline.progress()``.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

from torchrec.streamable import Pipelineable


class PipelineModelWrapper(nn.Module):
    """Delegates to a shared compute_loss callable under AMP autocast.

    Args:
        compute_loss_fn: callable(batch_dict) -> (total_loss, task_losses).
            Typically ``DistributedTrainer._compute_loss`` with active_tasks
            already bound via ``functools.partial`` or a lambda.
        amp_dtype: dtype for autocast (e.g. torch.bfloat16).
        use_amp: whether to enable autocast.
    """

    def __init__(
        self,
        compute_loss_fn: Callable,
        amp_dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
    ):
        super().__init__()
        self._compute_loss = compute_loss_fn
        self.amp_dtype = amp_dtype
        self.use_amp = use_amp

    def forward(
        self, batch: Pipelineable,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_dict = batch.to_dict()
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            return self._compute_loss(batch_dict)
