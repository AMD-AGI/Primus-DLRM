"""DLRM++ baseline model, config-driven.

All feature names, table definitions, and dense feature specs come from
``Config``.  No hardcoded feature names in this module.

DLRM architecture:
  User tower: first scalar embedding + all pooled sequence features + raw dense
  Item tower: remaining scalar embeddings + projected dense features
  Interaction: ConcatMLP / Dot / DCNv2
  Heads: one per task (from config.task_names)

Note: DLRM uses mean-pooled EmbeddingBagCollection for sequence features
(unlike OneTrans which uses unpooled EmbeddingCollection).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from primus_dlrm.config import Config
from primus_dlrm.models.base import BaseModel
from primus_dlrm.models.embedding import TorchRecEmbeddings
from primus_dlrm.models.interaction import build_interaction, _make_mlp


class DLRMBaseline(BaseModel):
    """Config-driven DLRM++ with split history pools.

    User tower: first scalar embedding + pooled sequence groups + raw dense.
    Item tower: remaining scalar embeddings + projected dense.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device | None = None,
        meta_device: bool = False,
    ):
        super().__init__()
        self._pipeline_mode = False
        self.config = config
        mc = config.model
        fc = config.feature
        self.tasks = list(config.task_names)
        D = mc.embedding_dim

        if device is None:
            device = torch.device("cpu")
        emb_device = torch.device("meta") if meta_device else device

        self.emb = TorchRecEmbeddings(
            mc.resolved_embedding_tables(),
            device=emb_device,
            embedding_init=mc.embedding_init,
            scalar_feature_names=set(fc.scalar_feature_names),
        )

        self.dense_projs = nn.ModuleDict()
        for df in fc.dense_features:
            if df.project:
                act: nn.Module = nn.ReLU(inplace=True) if df.activation == "relu" else nn.GELU()
                self.dense_projs[df.name] = nn.Sequential(
                    nn.Linear(df.dim, D, device=device), act,
                )

        n_seq_features = sum(len(f) for f in fc.sequence_groups.values())
        n_user_raw = sum(df.dim for df in fc.dense_features
                         if not df.project and df.in_user_tower())

        n_user_scalars = len(fc.user_scalar_features)
        user_feature_dim = n_user_scalars * D + n_seq_features * D + n_user_raw
        if mc.bottom_mlp_dims:
            self.user_bottom_mlp = _make_mlp(
                [user_feature_dim] + mc.bottom_mlp_dims, mc.dropout,
            ).to(device)
            user_out_dim = mc.bottom_mlp_dims[-1]
        else:
            self.user_bottom_mlp = nn.Identity()
            user_out_dim = user_feature_dim

        n_item_scalars = len(fc.item_scalar_features)
        n_proj_dense = sum(1 for df in fc.dense_features if df.project)
        n_item_raw = sum(df.dim for df in fc.dense_features
                         if not df.project and df.in_item_tower())
        item_feature_dim = (n_item_scalars + n_proj_dense) * D + n_item_raw
        if mc.bottom_mlp_dims:
            self.item_bottom_mlp = _make_mlp(
                [item_feature_dim] + mc.bottom_mlp_dims, mc.dropout,
            ).to(device)
            item_out_dim = mc.bottom_mlp_dims[-1]
        else:
            self.item_bottom_mlp = nn.Identity()
            item_out_dim = item_feature_dim

        item_dims_for_interaction = [item_out_dim]
        has_cross_proj = any(df.project and df.activation == "relu" for df in fc.dense_features)
        if has_cross_proj:
            item_dims_for_interaction.append(D)

        self.interaction = build_interaction(
            mc.interaction_type,
            user_dims=[user_out_dim],
            item_dims=item_dims_for_interaction,
            top_mlp_dims=mc.top_mlp_dims,
            dcn_num_cross_layers=mc.dcn_num_cross_layers,
            dropout=mc.dropout,
        ).to(device)

        self.heads = nn.ModuleDict({
            task: nn.Linear(self.interaction.output_dim, 1, device=device)
            for task in self.tasks
        })

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def _lookup_all(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Schema-driven lookup: maps batch keys to EC/EBC feature names."""
        feat_to_key = self.config.feature_to_batch_key()
        lookup_dict = {
            feat_name: batch[feat_to_key.get(feat_name, feat_name)]
            for feat_name in self.config.feature.all_ec_feature_names()
        }
        return self.emb(lookup_dict)

    def _lookup_pipeline(self, batch: dict) -> dict[str, torch.Tensor]:
        """Embedding lookup via pre-built KJT (pipeline mode)."""
        first_ec = self.config.feature.scalar_feature_names[0]
        first_key = self.config.feature_to_batch_key().get(first_ec, first_ec)
        B = batch[first_key].shape[0]

        embs: dict[str, torch.Tensor] = {}

        if self.emb.ec is not None:
            ec_out = self.emb.ec(batch["unpooled_kjt"])
            for feat in self.emb._unpooled_features:
                jt = ec_out[feat]
                emb = jt.values()
                D = self.emb._feature_dims[feat]
                if feat in self.emb._scalar_features:
                    embs[feat] = emb.view(B, D)
                else:
                    embs[feat] = emb.view(B, -1, D)

        if self.emb.ebc is not None:
            ebc_out = self.emb.ebc(batch["unpooled_kjt"])
            for feat in self.emb._pooled_features:
                embs[feat] = ebc_out[feat]

        return embs

    # ------------------------------------------------------------------
    # Towers
    # ------------------------------------------------------------------

    def _user_tower(
        self, embs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """User tower: user scalars + all pooled sequences + raw dense."""
        parts = [embs[sf.name] for sf in self.config.feature.user_scalar_features]
        for group_feats in self.config.feature.sequence_groups.values():
            for feat in group_feats:
                parts.append(embs[feat])
        for df in self.config.feature.dense_features:
            if not df.project and df.in_user_tower():
                parts.append(batch[df.name])
        return self.user_bottom_mlp(torch.cat(parts, dim=-1))

    def _item_tower(
        self, embs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Item tower: item scalars + projected dense + raw dense."""
        parts = [embs[sf.name] for sf in self.config.feature.item_scalar_features]
        for df in self.config.feature.dense_features:
            if df.project:
                parts.append(self.dense_projs[df.name](batch[df.name]))
        for df in self.config.feature.dense_features:
            if not df.project and df.in_item_tower():
                parts.append(batch[df.name])
        return self.item_bottom_mlp(torch.cat(parts, dim=-1))

    def _get_cross_features(
        self, batch: dict[str, torch.Tensor], item_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        feats = [item_features]
        for df in self.config.feature.dense_features:
            if df.project and df.activation == "relu":
                feats.append(self.dense_projs[df.name](batch[df.name]))
                break
        return feats

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self._pipeline_mode:
            embs = self._lookup_pipeline(batch)
        else:
            embs = self._lookup_all(batch)
        user_features = self._user_tower(embs, batch)
        item_features = self._item_tower(embs, batch)

        interaction_out = self.interaction(
            user_features=[user_features],
            item_features=self._get_cross_features(batch, item_features),
        )
        return {t: h(interaction_out).squeeze(-1) for t, h in self.heads.items()}

    def forward_with_cross_scores(
        self, batch: dict[str, torch.Tensor], cross_task: str = "",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        embs = self._lookup_all(batch)
        user_features = self._user_tower(embs, batch)
        item_features = self._item_tower(embs, batch)

        item_cross_parts = self._get_cross_features(batch, item_features)

        interaction_out = self.interaction(
            user_features=[user_features],
            item_features=item_cross_parts,
        )
        preds = {t: h(interaction_out).squeeze(-1) for t, h in self.heads.items()}

        B = user_features.size(0)
        user_exp = user_features.unsqueeze(1).expand(B, B, -1)
        item_exp = item_features.unsqueeze(0).expand(B, B, -1)

        cross_item_feats = [item_exp.reshape(B * B, -1)]
        for extra in item_cross_parts[1:]:
            D = extra.shape[-1]
            cross_item_feats.append(
                torch.zeros(B * B, D, device=extra.device, dtype=extra.dtype)
            )
        task = cross_task or self.tasks[0]
        cross_interaction = self.interaction(
            user_features=[user_exp.reshape(B * B, -1)],
            item_features=cross_item_feats,
        )
        cross_scores = self.heads[task](cross_interaction).squeeze(-1).view(B, B)
        return preds, cross_scores