"""Runtime model+loss wrapper for TrainPipelineSparseDist.

TrainPipelineSparseDist needs two separate callables:

  1. The *DMP model* itself — FX-traced to discover which inputs feed into
     EmbeddingCollection so the pipeline can schedule input_dist / output_dist
     on dedicated CUDA streams.  This is the ``model`` arg.

  2. A *custom forward* callable — invoked on the default stream once the
     redistributed embeddings are ready.  This is the ``custom_model_fwd`` arg
     and is what this module provides.

Why two?  FX tracing must see a clean DlrmBatch → EC path (controlled by
``_pipeline_mode``), but at runtime we also need AMP autocast, loss
computation, and optional contrastive loss — none of which should be traced.
This wrapper keeps runtime-only logic out of the FX graph.

The pipeline calls  ``custom_model_fwd(batch)``  on the default stream during
Stage 2 of each step.  It expects the return value's first element to be the
scalar loss (used by ``backward()``); any additional elements are forwarded as
the return value of ``pipeline.progress()``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from primus_dlrm.data.pipeline_batch import DlrmBatch
from primus_dlrm.training.losses import MultiTaskLoss, InBatchBPRLoss


class PipelineModelWrapper(nn.Module):
    """Bundles DMP model forward + loss into a single callable for the pipeline.

    Registered as ``custom_model_fwd`` in TrainPipelineSparseDist.  The
    pipeline invokes ``self(batch)`` after the embedding output_dist for
    this batch has completed (All2All_Seq_fwd finished, redistributed
    embeddings available on every rank).

    Returns ``(total_loss, task_losses)`` — the pipeline calls ``.backward()``
    on ``total_loss`` and passes ``task_losses`` back from ``progress()`` for
    logging.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultiTaskLoss,
        active_tasks: list[str],
        contrastive_loss_fn: Optional[InBatchBPRLoss] = None,
        contrastive_weight: float = 0.0,
        amp_dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.active_tasks = active_tasks
        self.contrastive_loss_fn = contrastive_loss_fn
        self.contrastive_weight = contrastive_weight
        self.amp_dtype = amp_dtype
        self.use_amp = use_amp

    def forward(
        self, batch: DlrmBatch,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # DlrmBatch is a Pipelineable dataclass; convert to a plain dict
        # so the model's forward() can do batch["item_id"] etc.
        batch_dict = batch.to_dict()

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            if self.contrastive_weight > 0 and self.contrastive_loss_fn is not None:
                # Contrastive mode: compute both task predictions and
                # cross-item similarity scores for BPR loss.  Must unwrap
                # DDP/DMP to call the non-standard forward_with_cross_scores.
                inner = self.model.module if hasattr(self.model, "module") else self.model
                preds, cross_scores = inner.forward_with_cross_scores(batch_dict)
            else:
                # Standard mode: DMP's __call__ handles the sharded embedding
                # lookup + dense forward.  At this point the embeddings have
                # already been redistributed by the pipeline's output_dist, so
                # the model receives fully-gathered embedding vectors.
                preds = self.model(batch_dict)
                cross_scores = None

            # Extract ground-truth labels for the active tasks (e.g.
            # listen_plus, like, dislike) from the same batch dict.
            labels = {t: batch_dict[t] for t in self.active_tasks}
            total_loss, task_losses = self.loss_fn(preds, labels)

            if cross_scores is not None:
                # In-batch BPR contrastive loss: encourages the model to rank
                # positive items higher than negatives sampled from the batch.
                bpr_loss = self.contrastive_loss_fn(
                    cross_scores, batch_dict["listen_plus"],
                )
                total_loss = total_loss + self.contrastive_weight * bpr_loss
                task_losses["bpr"] = bpr_loss

        # total_loss  → pipeline calls .backward() on this
        # task_losses → returned by pipeline.progress() for logging
        return total_loss, task_losses
