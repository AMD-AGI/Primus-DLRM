"""DLRM++ baseline model, schema-driven.

All feature names, table definitions, and dense feature specs come from a
``FeatureSchema``.  No hardcoded feature names in this module.

DLRM architecture:
  User tower: first scalar embedding + all pooled sequence features + raw dense
  Item tower: remaining scalar embeddings + projected dense features
  Interaction: ConcatMLP / Dot / DCNv2
  Heads: one per task (from schema.task_names)

Note: DLRM uses mean-pooled EmbeddingBagCollection for sequence features
(unlike OneTrans which uses unpooled EmbeddingCollection).  The schema's
embedding tables are cloned with ``pooling="mean"`` for sequence features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from primus_dlrm.config import ModelConfig
from primus_dlrm.models.base import BaseModel
from primus_dlrm.models.embedding import TableSpec, TorchRecEmbeddings
from primus_dlrm.models.interaction import build_interaction, _make_mlp
from primus_dlrm.schema import FeatureSchema


class DLRMBaseline(BaseModel):
    """Schema-driven DLRM++ with split history pools.

    User tower: first scalar embedding + pooled sequence groups + raw dense.
    Item tower: remaining scalar embeddings + projected dense.
    """

    def __init__(
        self,
        config: ModelConfig,
        schema: FeatureSchema,
        device: torch.device | None = None,
        meta_device: bool = False,
    ):
        super().__init__()
        self._pipeline_mode = False
        self.config = config
        self.schema = schema
        self.tasks = list(schema.task_names)
        D = schema.embedding_dim

        if device is None:
            device = torch.device("cpu")
        emb_device = torch.device("meta") if meta_device else device

        # Build DLRM-specific table specs: pooled (mean) for sequence features,
        # unpooled for scalar features.
        dlrm_tables = self._build_dlrm_tables(schema, D)
        self.emb = TorchRecEmbeddings(
            dlrm_tables,
            device=emb_device,
            embedding_init=schema.embedding_init,
            scalar_feature_names=set(schema.scalar_features),
        )

        # Dense projections (same as OneTrans: project=True gets Linear+act)
        self.dense_projs = nn.ModuleDict()
        for df in schema.dense_features:
            if df.project:
                act: nn.Module = nn.ReLU(inplace=True) if df.activation == "relu" else nn.GELU()
                self.dense_projs[df.name] = nn.Sequential(
                    nn.Linear(df.dim, D, device=device), act,
                )

        # All sequence features (pooled) produce D-dim outputs
        n_seq_features = sum(len(f) for f in schema.sequence_groups.values())
        n_raw_dense = sum(df.dim for df in schema.dense_features if not df.project)

        # User tower: 1 scalar (first) + all pooled sequences + raw dense
        user_feature_dim = D + n_seq_features * D + n_raw_dense
        if config.bottom_mlp_dims:
            self.user_bottom_mlp = _make_mlp(
                [user_feature_dim] + config.bottom_mlp_dims, config.dropout,
            ).to(device)
            user_out_dim = config.bottom_mlp_dims[-1]
        else:
            self.user_bottom_mlp = nn.Identity()
            user_out_dim = user_feature_dim

        # Item tower: remaining scalars + projected dense
        n_item_scalars = len(schema.scalar_features) - 1
        n_proj_dense = sum(1 for df in schema.dense_features if df.project)
        item_feature_dim = (n_item_scalars + n_proj_dense) * D + n_raw_dense
        if config.bottom_mlp_dims:
            self.item_bottom_mlp = _make_mlp(
                [item_feature_dim] + config.bottom_mlp_dims, config.dropout,
            ).to(device)
            item_out_dim = config.bottom_mlp_dims[-1]
        else:
            self.item_bottom_mlp = nn.Identity()
            item_out_dim = item_feature_dim

        # Cross-feature projection (projected dense features for interaction)
        item_dims_for_interaction = [item_out_dim]
        has_cross_proj = any(df.project and df.activation == "relu" for df in schema.dense_features)
        if has_cross_proj:
            item_dims_for_interaction.append(D)

        self.interaction = build_interaction(
            config.interaction_type,
            user_dims=[user_out_dim],
            item_dims=item_dims_for_interaction,
            top_mlp_dims=config.top_mlp_dims,
            dcn_num_cross_layers=config.dcn_num_cross_layers,
            dropout=config.dropout,
        ).to(device)

        self.heads = nn.ModuleDict({
            task: nn.Linear(self.interaction.output_dim, 1, device=device)
            for task in self.tasks
        })

    @staticmethod
    def _build_dlrm_tables(schema: FeatureSchema, D: int) -> list[TableSpec]:
        """Build DLRM table specs from schema.

        Sequence features use pooled EBC (pooling type from schema.pooling),
        scalar features use unpooled EC.
        """
        seq_feature_set = set()
        for feats in schema.sequence_groups.values():
            seq_feature_set.update(feats)

        tables: list[TableSpec] = []
        for orig in schema.embedding_tables:
            pooled_feats = [f for f in orig.feature_names if f in seq_feature_set]
            unpooled_feats = [f for f in orig.feature_names if f not in seq_feature_set]

            if pooled_feats:
                tables.append(TableSpec(
                    name=f"hist_{orig.name}",
                    num_embeddings=orig.num_embeddings,
                    embedding_dim=D,
                    pooling=schema.pooling,
                    feature_names=pooled_feats,
                ))
            if unpooled_feats:
                tables.append(TableSpec(
                    name=orig.name,
                    num_embeddings=orig.num_embeddings,
                    embedding_dim=D,
                    pooling="none",
                    feature_names=unpooled_feats,
                ))

        return tables

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def _lookup_all(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Schema-driven lookup: maps batch keys to EC/EBC feature names."""
        feat_to_key = self.schema.feature_to_batch_key()
        lookup_dict = {
            feat_name: batch[feat_to_key.get(feat_name, feat_name)]
            for feat_name in self.schema.all_ec_feature_names()
        }
        return self.emb(lookup_dict)

    def _lookup_pipeline(self, batch: dict) -> dict[str, torch.Tensor]:
        """Embedding lookup via pre-built KJT (pipeline mode)."""
        first_ec = self.schema.scalar_features[0]
        first_key = self.schema.feature_to_batch_key().get(first_ec, first_ec)
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
        """User tower: first scalar + all pooled sequences + raw dense."""
        parts = [embs[self.schema.scalar_features[0]]]
        for group_feats in self.schema.sequence_groups.values():
            for feat in group_feats:
                parts.append(embs[feat])
        for df in self.schema.dense_features:
            if not df.project:
                parts.append(batch[df.name])
        return self.user_bottom_mlp(torch.cat(parts, dim=-1))

    def _item_tower(
        self, embs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Item tower: remaining scalars + projected dense + raw dense."""
        parts = [embs[f] for f in self.schema.scalar_features[1:]]
        for df in self.schema.dense_features:
            if df.project:
                parts.append(self.dense_projs[df.name](batch[df.name]))
        for df in self.schema.dense_features:
            if not df.project:
                parts.append(batch[df.name])
        return self.item_bottom_mlp(torch.cat(parts, dim=-1))

    def _get_cross_features(
        self, batch: dict[str, torch.Tensor], item_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        feats = [item_features]
        for df in self.schema.dense_features:
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

        interaction_out = self.interaction(
            user_features=[user_features],
            item_features=self._get_cross_features(batch, item_features),
        )
        preds = {t: h(interaction_out).squeeze(-1) for t, h in self.heads.items()}

        B = user_features.size(0)
        user_exp = user_features.unsqueeze(1).expand(B, B, -1)
        item_exp = item_features.unsqueeze(0).expand(B, B, -1)

        cross_item_feats = [item_exp.reshape(B * B, -1)]
        task = cross_task or self.tasks[0]
        cross_interaction = self.interaction(
            user_features=[user_exp.reshape(B * B, -1)],
            item_features=cross_item_feats,
        )
        cross_scores = self.heads[task](cross_interaction).squeeze(-1).view(B, B)
        return preds, cross_scores
