"""TorchRec embedding tables with tensor-in/tensor-out API.

Provides TorchRecEmbeddings: a unified module wrapping EmbeddingBagCollection
(pooled) and EmbeddingCollection (unpooled). DistributedModelParallel discovers
the EBC/EC submodules automatically for sharding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
)
from torchrec.modules.embedding_configs import PoolingType


@dataclass
class TableSpec:
    """One physical embedding table, possibly serving multiple feature names."""
    name: str
    num_embeddings: int
    embedding_dim: int
    pooling: str = "none"  # "mean", "sum", "none"
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = [self.name]


_POOLING_MAP = {"mean": PoolingType.MEAN, "sum": PoolingType.SUM}


def _build_torchrec_configs(
    table_specs: list[TableSpec],
    embedding_init: str = "uniform",
) -> tuple[list[EmbeddingBagConfig], list[EmbeddingConfig], list[str], list[str], dict[str, int]]:
    """Build TorchRec EBC/EC configs from TableSpecs.

    Returns (pooled_configs, unpooled_configs, pooled_features, unpooled_features, feature_dims).
    """
    init_fn = None
    if embedding_init == "normal":
        def init_fn(w: torch.Tensor) -> None:
            nn.init.normal_(w, mean=0.0, std=1.0)

    pooled_configs: list[EmbeddingBagConfig] = []
    unpooled_configs: list[EmbeddingConfig] = []
    pooled_features: list[str] = []
    unpooled_features: list[str] = []
    feature_dims: dict[str, int] = {}

    for spec in table_specs:
        for feat in spec.feature_names:
            feature_dims[feat] = spec.embedding_dim

        extra = {"init_fn": init_fn} if init_fn is not None else {}
        if spec.pooling in _POOLING_MAP:
            pooled_configs.append(EmbeddingBagConfig(
                name=spec.name,
                embedding_dim=spec.embedding_dim,
                num_embeddings=spec.num_embeddings,
                feature_names=list(spec.feature_names),
                pooling=_POOLING_MAP[spec.pooling],
                **extra,
            ))
            pooled_features.extend(spec.feature_names)
        else:
            unpooled_configs.append(EmbeddingConfig(
                name=spec.name,
                embedding_dim=spec.embedding_dim,
                num_embeddings=spec.num_embeddings,
                feature_names=list(spec.feature_names),
                **extra,
            ))
            unpooled_features.extend(spec.feature_names)

    return pooled_configs, unpooled_configs, pooled_features, unpooled_features, feature_dims


class TorchRecEmbeddings(nn.Module):
    """Unified embedding module wrapping TorchRec EBC + EC.

    Usage::

        emb = TorchRecEmbeddings(table_specs, device)
        results = emb({
            "uid": batch["uid"],           # [B]   -> [B, D]
            "item": batch["item_id"],      # [B]   -> [B, D]
            "hist_lp_item": batch["..."],  # [B,L] -> [B, D] (pooled) or [B,L,D] (unpooled)
        })

    DistributedModelParallel discovers ``self.ebc`` / ``self.ec`` for sharding.

    When ``device`` is ``torch.device("meta")``, tables are created on the meta
    device (no memory allocated). This is required for DMP, which materializes
    weights on the correct GPU per the sharding plan.
    """

    def __init__(
        self,
        table_specs: list[TableSpec],
        device: torch.device | None = None,
        embedding_init: str = "uniform",
        scalar_feature_names: set[str] | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        (pooled_configs, unpooled_configs,
         self._pooled_features, self._unpooled_features,
         self._feature_dims) = _build_torchrec_configs(table_specs, embedding_init)

        self._pooled_set = set(self._pooled_features)
        self._unpooled_set = set(self._unpooled_features)
        # Scalar (1D) vs sequence (2D) — avoids dim() checks that break FX tracing.
        if scalar_feature_names is not None:
            self._scalar_features: set[str] = scalar_feature_names & self._unpooled_set
        else:
            self._scalar_features: set[str] = {
                f for f in self._unpooled_features if not f.startswith("hist_")
            }

        self.ebc: EmbeddingBagCollection | None = None
        self.ec: EmbeddingCollection | None = None

        if pooled_configs:
            self.ebc = EmbeddingBagCollection(
                tables=pooled_configs, device=device,
            )
        if unpooled_configs:
            self.ec = EmbeddingCollection(
                tables=unpooled_configs, device=device,
            )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        padding_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Batch-lookup all features.

        Args:
            features: feature_name -> ID tensor.
                Pooled features: ``[B, L]`` -> ``[B, D]`` (mean/sum).
                Unpooled features: ``[B]`` -> ``[B, D]``;
                                   ``[B, L]`` -> ``[B, L, D]``.
            padding_idx: ID to exclude from pooled lookups (matches
                ``nn.EmbeddingBag(padding_idx=0)`` behaviour).

        Returns:
            dict mapping each feature_name to its embedding tensor.
        """
        results: dict[str, torch.Tensor] = {}

        # --- pooled (EBC) ---
        if self.ebc is not None:
            pooled = {k: v for k, v in features.items() if k in self._pooled_set}
            if pooled:
                kjt = _build_pooled_kjt(pooled, self._pooled_features, padding_idx)
                kt = self.ebc(kjt)
                for feat in self._pooled_features:
                    if feat in pooled:
                        results[feat] = kt[feat]

        # --- unpooled (EC) ---
        if self.ec is not None:
            unpooled = {k: v for k, v in features.items() if k in self._unpooled_set}
            if unpooled:
                kjt = _build_unpooled_kjt(unpooled, self._unpooled_features)
                ec_out = self.ec(kjt)
                for feat in self._unpooled_features:
                    if feat not in unpooled:
                        continue
                    jt = ec_out[feat]
                    emb = jt.values()
                    D = self._feature_dims[feat]
                    B = unpooled[feat].shape[0]
                    if feat in self._scalar_features:
                        results[feat] = emb.view(B, D)
                    else:
                        results[feat] = emb.view(B, -1, D)

        return results


# ---- KJT builders (module-level helpers) ----

def _build_pooled_kjt(
    features: dict[str, torch.Tensor],
    feature_order: list[str],
    padding_idx: int,
) -> KeyedJaggedTensor:
    keys: list[str] = []
    all_values: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    for name in feature_order:
        if name not in features:
            continue
        ids = features[name]  # [B, L]
        mask = ids != padding_idx
        lengths = mask.sum(dim=1).to(torch.int32)
        values = ids[mask]

        keys.append(name)
        all_values.append(values)
        all_lengths.append(lengths)

    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.cat(all_values) if all_values else torch.empty(0, dtype=torch.long),
        lengths=torch.cat(all_lengths) if all_lengths else torch.empty(0, dtype=torch.int32),
    )


def _build_unpooled_kjt(
    features: dict[str, torch.Tensor],
    feature_order: list[str],
) -> KeyedJaggedTensor:
    keys: list[str] = []
    all_values: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    for name in feature_order:
        if name not in features:
            continue
        ids = features[name]
        B = ids.shape[0]
        flat = ids.reshape(-1)
        L = flat.shape[0] // B
        keys.append(name)
        all_values.append(flat)
        all_lengths.append(
            torch.full((B,), L, dtype=torch.int32, device=ids.device),
        )

    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.cat(all_values) if all_values else torch.empty(0, dtype=torch.long),
        lengths=torch.cat(all_lengths) if all_lengths else torch.empty(0, dtype=torch.int32),
    )
