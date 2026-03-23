"""Base model interface for the benchmark suite."""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base interface for all recommendation models.

    Flat-batch interface: each sample is one (user, item) scoring pair.
    Forward returns a dict of task_name -> [B] logits/predictions.
    """

    @abstractmethod
    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ...

    def forward_with_cross_scores(
        self, batch: dict[str, torch.Tensor], cross_task: str = "listen_plus",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Forward that also returns a [B,B] cross-batch contrastive score matrix.

        This is a separate method (not a flag on forward()) because TorchRec's
        TrainPipelineSparseDist runs torch.fx.symbolic_trace on forward().
        FX creates a Proxy for every forward() parameter not listed in
        concrete_args.  A keyword arg like ``return_cross_scores=False``
        becomes a Proxy, and ``if not return_cross_scores`` then calls
        ``Proxy.__bool__`` which raises ``TraceError: symbolically traced
        variables cannot be used as inputs to control flow``.

        Keeping forward() signature to just ``(self, batch)`` avoids this.
        The contrastive path is only called from PipelineModelWrapper and the
        sequential trainer, neither of which is FX-traced.
        """
        raise NotImplementedError
