"""Multi-task prediction heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Produces per-task predictions from interaction output."""

    def __init__(self, input_dim: int, tasks: list[str] | None = None):
        super().__init__()
        if tasks is None:
            tasks = ["listen_plus", "like", "dislike", "played_ratio"]
        self.tasks = tasks
        self.heads = nn.ModuleDict({
            task: nn.Linear(input_dim, 1) for task in tasks
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, D_out] interaction features.

        Returns:
            Dict of task_name -> [B] raw logits (for classification)
            or predictions (for regression).
        """
        return {task: head(x).squeeze(-1) for task, head in self.heads.items()}
