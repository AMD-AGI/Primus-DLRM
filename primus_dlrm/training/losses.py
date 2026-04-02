"""Configurable multi-task loss for DLRM++."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Weighted combination of per-task losses.

    Active tasks and their weights are driven by the weights dict.
    Tasks in ``regression_tasks`` use MSE loss; all others use BCE with logits.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        regression_tasks: list[str] | None = None,
    ):
        super().__init__()
        if weights is None:
            weights = {"task0": 1.0}
        self.weights = {k: v for k, v in weights.items() if v > 0}
        self._regression_tasks: set[str] = set(regression_tasks or [])

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        task_losses = {}

        for task, weight in self.weights.items():
            if task not in predictions or task not in labels:
                continue
            if task in self._regression_tasks:
                task_losses[task] = F.mse_loss(predictions[task], labels[task])
            else:
                task_losses[task] = F.binary_cross_entropy_with_logits(
                    predictions[task], labels[task]
                )

        total = sum(self.weights[k] * v for k, v in task_losses.items())
        return total, task_losses


class InBatchBPRLoss(nn.Module):
    """In-batch BPR contrastive loss.

    For each positive anchor (user with label=1), treats all other items in
    the batch as negatives and optimizes pairwise ranking:
        loss = -log(sigmoid(s_pos - s_neg))

    Requires a [B, B] cross-score matrix where entry (i, j) is the model
    score for (user_i, item_j).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        cross_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cross_scores: [B, B] score matrix, entry (i,j) = score(user_i, item_j).
            labels: [B] binary labels (1 = positive anchor, 0 = skip).

        Returns:
            Scalar BPR loss averaged over positive anchors.
        """
        B = cross_scores.size(0)
        pos_mask = labels.bool()
        num_pos = pos_mask.sum().item()
        if num_pos == 0:
            return cross_scores.new_tensor(0.0)

        pos_scores = cross_scores[pos_mask]  # [P, B]
        diag_indices = torch.where(pos_mask)[0]  # indices of positives in original batch
        pos_self = pos_scores[torch.arange(num_pos, device=cross_scores.device), diag_indices]  # [P]

        neg_mask = torch.ones(num_pos, B, dtype=torch.bool, device=cross_scores.device)
        neg_mask[torch.arange(num_pos, device=cross_scores.device), diag_indices] = False

        diff = (pos_self.unsqueeze(1) - pos_scores) / self.temperature  # [P, B]
        diff = diff.masked_fill(~neg_mask, 1e9)

        loss = -F.logsigmoid(diff)
        loss = loss.masked_fill(~neg_mask, 0.0)
        loss = loss.sum() / (num_pos * (B - 1))
        return loss
