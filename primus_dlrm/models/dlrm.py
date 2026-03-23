"""DLRM++ baseline model with split history pools, backed by TorchRec embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn

from primus_dlrm.config import ModelConfig
from primus_dlrm.models.base import BaseModel
from primus_dlrm.models.embedding import TableSpec, TorchRecEmbeddings
from primus_dlrm.models.interaction import build_interaction, _make_mlp


def _dlrm_table_specs(
    num_users: int, num_items: int, num_artists: int, num_albums: int,
    embedding_dim: int,
) -> list[TableSpec]:
    """Table specs for DLRM: 3 pooled history tables + 4 unpooled tables."""
    return [
        # Pooled (EBC) -- one physical table per entity, 3 features each
        TableSpec("hist_item", num_items, embedding_dim, "mean",
                  ["hist_lp_item", "hist_like_item", "hist_skip_item"]),
        TableSpec("hist_artist", num_artists, embedding_dim, "mean",
                  ["hist_lp_artist", "hist_like_artist", "hist_skip_artist"]),
        TableSpec("hist_album", num_albums, embedding_dim, "mean",
                  ["hist_lp_album", "hist_like_album", "hist_skip_album"]),
        # Unpooled (EC)
        TableSpec("uid", num_users, embedding_dim),
        TableSpec("item", num_items, embedding_dim),
        TableSpec("artist", num_artists, embedding_dim),
        TableSpec("album", num_albums, embedding_dim),
    ]


class DLRMBaseline(BaseModel):
    """DLRM++ with split history pools (listen+, like, skip).

    User tower: uid embedding + 3 pools x (item + artist + album) mean-pooled.
    Item tower: item/artist/album embeddings + audio projection.
    Interaction: ConcatMLP / Dot / DCNv2.
    Heads: configurable via tasks list.

    All embeddings managed by TorchRecEmbeddings (EBC + EC).
    """

    def __init__(
        self,
        config: ModelConfig,
        num_users: int,
        num_items: int,
        num_artists: int,
        num_albums: int,
        audio_input_dim: int = 256,
        device: torch.device | None = None,
        tasks: list[str] | None = None,
        num_counter_windows: int = 0,
        meta_device: bool = False,
    ):
        super().__init__()
        self.config = config
        self.tasks = tasks or ["listen_plus"]
        self.num_counter_windows = num_counter_windows
        D = config.embedding_dim

        if device is None:
            device = torch.device("cpu")

        emb_device = torch.device("meta") if meta_device else device
        self.emb = TorchRecEmbeddings(
            _dlrm_table_specs(num_users, num_items, num_artists, num_albums, D),
            device=emb_device,
            embedding_init=config.embedding_init,
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_input_dim, D, device=device),
            nn.ReLU(inplace=True),
        )

        # User: uid[D] + 3 pools x 3 features x D = 10D + optional 3*W counters
        user_feature_dim = 10 * D + 3 * num_counter_windows
        if config.bottom_mlp_dims:
            self.user_bottom_mlp = _make_mlp(
                [user_feature_dim] + config.bottom_mlp_dims, config.dropout
            ).to(device)
            user_out_dim = config.bottom_mlp_dims[-1]
        else:
            self.user_bottom_mlp = nn.Identity()
            user_out_dim = user_feature_dim

        # Item: item[D] + artist[D] + album[D] + audio[D] = 4D + optional 3*W counters
        item_feature_dim = 4 * D + 3 * num_counter_windows
        if config.bottom_mlp_dims:
            self.item_bottom_mlp = _make_mlp(
                [item_feature_dim] + config.bottom_mlp_dims, config.dropout
            ).to(device)
            item_out_dim = config.bottom_mlp_dims[-1]
        else:
            self.item_bottom_mlp = nn.Identity()
            item_out_dim = item_feature_dim

        item_dims_for_interaction = [item_out_dim]
        if num_counter_windows > 0:
            self.cross_proj = nn.Sequential(
                nn.Linear(9 * num_counter_windows, D, device=device),
                nn.ReLU(inplace=True),
            )
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

    # ------------------------------------------------------------------
    # Embedding lookup helpers
    # ------------------------------------------------------------------

    def _lookup_all(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Single batched TorchRec lookup for all features."""
        return self.emb({
            "hist_lp_item": batch["hist_lp_item_ids"],
            "hist_like_item": batch["hist_like_item_ids"],
            "hist_skip_item": batch["hist_skip_item_ids"],
            "hist_lp_artist": batch["hist_lp_artist_ids"],
            "hist_like_artist": batch["hist_like_artist_ids"],
            "hist_skip_artist": batch["hist_skip_artist_ids"],
            "hist_lp_album": batch["hist_lp_album_ids"],
            "hist_like_album": batch["hist_like_album_ids"],
            "hist_skip_album": batch["hist_skip_album_ids"],
            "uid": batch["uid"],
            "item": batch["item_id"],
            "artist": batch["artist_id"],
            "album": batch["album_id"],
        })

    # ------------------------------------------------------------------
    # Towers
    # ------------------------------------------------------------------

    def _get_item_features_list(
        self, batch: dict[str, torch.Tensor], item_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        feats = [item_features]
        if self.num_counter_windows > 0 and "cross_counters" in batch:
            feats.append(self.cross_proj(batch["cross_counters"]))
        return feats

    def _user_tower(
        self, embs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = [
            embs["uid"],
            embs["hist_lp_item"], embs["hist_lp_artist"], embs["hist_lp_album"],
            embs["hist_like_item"], embs["hist_like_artist"], embs["hist_like_album"],
            embs["hist_skip_item"], embs["hist_skip_artist"], embs["hist_skip_album"],
        ]
        if self.num_counter_windows > 0 and "user_counters" in batch:
            parts.append(batch["user_counters"])
        return self.user_bottom_mlp(torch.cat(parts, dim=-1))

    def _item_tower(
        self, embs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = [embs["item"], embs["artist"], embs["album"],
                 self.audio_proj(batch["audio_embed"])]
        if self.num_counter_windows > 0 and "item_counters" in batch:
            parts.append(batch["item_counters"])
        return self.item_bottom_mlp(torch.cat(parts, dim=-1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        embs = self._lookup_all(batch)
        user_features = self._user_tower(embs, batch)
        item_features = self._item_tower(embs, batch)

        interaction_out = self.interaction(
            user_features=[user_features],
            item_features=self._get_item_features_list(batch, item_features),
        )
        preds = {t: h(interaction_out).squeeze(-1) for t, h in self.heads.items()}
        return preds

    def forward_with_cross_scores(
        self, batch: dict[str, torch.Tensor], cross_task: str = "listen_plus",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        embs = self._lookup_all(batch)
        user_features = self._user_tower(embs, batch)
        item_features = self._item_tower(embs, batch)

        interaction_out = self.interaction(
            user_features=[user_features],
            item_features=self._get_item_features_list(batch, item_features),
        )
        preds = {t: h(interaction_out).squeeze(-1) for t, h in self.heads.items()}

        B = user_features.size(0)
        user_exp = user_features.unsqueeze(1).expand(B, B, -1)
        item_exp = item_features.unsqueeze(0).expand(B, B, -1)

        cross_item_feats = [item_exp.reshape(B * B, -1)]
        if self.num_counter_windows > 0 and "cross_counters" in batch:
            D = self.config.embedding_dim
            zeros = torch.zeros(B * B, D, device=user_features.device, dtype=user_features.dtype)
            cross_item_feats.append(zeros)

        cross_interaction = self.interaction(
            user_features=[user_exp.reshape(B * B, -1)],
            item_features=cross_item_feats,
        )
        cross_scores = self.heads[cross_task](cross_interaction).squeeze(-1).view(B, B)
        return preds, cross_scores
