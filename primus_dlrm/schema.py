"""FeatureSchema: domain-agnostic description of model inputs.

A FeatureSchema describes embedding tables, sparse features (grouped into
sequence pools and scalar features), dense features, and task labels.
The same schema drives both the model architecture (which features to look up,
how to group them) and the data pipeline (what tensors to generate or collate).

All feature names are defined in YAML config — no hardcoded names in Python.

Two builders are provided:
  - ``build_schema_from_config``: reads the ``data.schema`` section of a
    ``Config`` and combines it with runtime vocab sizes.
  - ``build_schema_from_synthetic``: auto-generates anonymous table/feature
    names from a ``SyntheticDataConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from primus_dlrm.config import (
    Config,
    DenseFeatureSpec,
    ModelConfig,
    SyntheticDataConfig,
)
from primus_dlrm.models.embedding import TableSpec


@dataclass
class FeatureSchema:
    """Domain-agnostic description of all model features."""

    embedding_tables: list[TableSpec]

    # Sequence feature groups: group_name -> list of EC feature names.
    # Each group is tokenized together (concatenated per position).
    sequence_groups: dict[str, list[str]] = field(default_factory=dict)

    # Scalar embedding features (single ID per sample, not sequences).
    scalar_features: list[str] = field(default_factory=list)

    # Dense (non-embedding) features.
    dense_features: list[DenseFeatureSpec] = field(default_factory=list)

    # Maps batch dict key -> EC feature name.
    # Only needed when they differ (e.g. batch["item_id"] -> EC "item").
    # If a feature is not in this dict, batch key == feature name.
    batch_to_feature: dict[str, str] = field(default_factory=dict)

    # Explicit KJT feature order for pipeline mode.  When set, KJT keys
    # are emitted in this order (must match EC's expected feature order).
    # When empty, falls back to all_ec_feature_names().
    kjt_feature_order: list[str] = field(default_factory=list)

    sequence_length: int = 20
    pooling: str = "mean"
    num_tasks: int = 1
    task_names: list[str] = field(default_factory=lambda: ["task0"])
    embedding_dim: int = 64
    embedding_init: str = "uniform"

    def all_ec_feature_names(self) -> list[str]:
        """All EC feature names in a deterministic order (scalars first)."""
        names = list(self.scalar_features)
        for feats in self.sequence_groups.values():
            names.extend(feats)
        return names

    def feature_to_batch_key(self) -> dict[str, str]:
        """Inverse of batch_to_feature: EC feature name -> batch key."""
        inv = {v: k for k, v in self.batch_to_feature.items()}
        for name in self.all_ec_feature_names():
            if name not in inv:
                inv[name] = name
        return inv


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------

def build_schema_from_config(
    config: Config,
    vocab_sizes: list[int],
) -> FeatureSchema:
    """Build a FeatureSchema from ``config.data.schema`` + runtime vocab sizes.

    All feature names, table definitions, sequence groups, dense features,
    and batch-to-feature mappings are read from the YAML config.  Only the
    vocabulary sizes (discovered at runtime from the dataset) are passed
    separately as a **positional list** matching the table order in the YAML::

        schema = build_schema_from_config(config, [
            dataset.num_items,    # matches 1st table in YAML
            dataset.num_artists,  # matches 2nd table
            dataset.num_albums,   # matches 3rd table
            num_users,            # matches 4th table
        ])

    Args:
        config: Top-level ``Config``.  The ``data.schema`` section defines
            tables, features, groups, and dense specs.  ``model`` provides
            ``embedding_dim`` and ``embedding_init``.  ``train.loss_weights``
            determines task names.
        vocab_sizes: Vocabulary sizes in the same order as
            ``config.data.schema.embedding_tables``.
    """
    dc = config.data
    mc = config.model
    tc = config.train
    sc = dc.schema
    D = mc.embedding_dim

    if not sc.embedding_tables:
        raise ValueError(
            "config.data.schema.embedding_tables is empty. "
            "Define embedding tables in the YAML config under data.schema."
        )

    tables = [
        TableSpec(
            name=t.name,
            num_embeddings=vocab_sizes[i] if i < len(vocab_sizes) else 0,
            embedding_dim=D,
            pooling="none",
            feature_names=list(t.features),
        )
        for i, t in enumerate(sc.embedding_tables)
    ]

    active_tasks = [k for k, v in tc.loss_weights.items() if v > 0]
    if not active_tasks:
        active_tasks = ["task0"]

    return FeatureSchema(
        embedding_tables=tables,
        sequence_groups=dict(sc.sequence_groups),
        scalar_features=list(sc.scalar_features),
        dense_features=list(sc.dense_features),
        batch_to_feature=dict(sc.batch_to_feature),
        kjt_feature_order=list(sc.kjt_feature_order),
        sequence_length=dc.history_length,
        pooling=sc.pooling,
        num_tasks=len(active_tasks),
        task_names=active_tasks,
        embedding_dim=D,
        embedding_init=mc.embedding_init,
    )


def build_schema_from_synthetic(
    syn: SyntheticDataConfig,
    model_config: ModelConfig,
    history_length: int = 20,
) -> FeatureSchema:
    """Auto-generate a FeatureSchema from a SyntheticDataConfig.

    Table and feature names are anonymous: ``t0``, ``t1``, ``f_t0_seq0``,
    ``f_t0_s0``, ``dense_0``, ``task0``, etc.
    """
    emb_dim = model_config.embedding_dim

    tables: list[TableSpec] = []
    sequence_groups: dict[str, list[str]] = {}
    scalar_features: list[str] = []

    # Track which features belong to each table
    table_feature_names: list[list[str]] = [[] for _ in syn.embedding_tables]

    group_counter = 0
    for spec in syn.sparse_features:
        ti = spec.table_index
        t_spec = syn.embedding_tables[ti]
        t_name = f"t{ti}"

        if spec.is_sequence:
            for fi in range(spec.num_features):
                feat_name = f"f_{t_name}_seq{fi}_{group_counter}"
                table_feature_names[ti].append(feat_name)
                group_name = f"group_{group_counter}"
                sequence_groups.setdefault(group_name, []).append(feat_name)
            group_counter += 1
        else:
            for fi in range(spec.num_features):
                feat_name = f"f_{t_name}_s{fi}_{len(scalar_features)}"
                table_feature_names[ti].append(feat_name)
                scalar_features.append(feat_name)

    for ti, t_spec in enumerate(syn.embedding_tables):
        tables.append(TableSpec(
            name=f"t{ti}",
            num_embeddings=t_spec.num_embeddings,
            embedding_dim=emb_dim,
            pooling="none",
            feature_names=table_feature_names[ti],
        ))

    # Remove empty tables (no features assigned)
    tables = [t for t in tables if t.feature_names]

    dense = list(syn.dense_features)
    task_names = [f"task{i}" for i in range(syn.num_tasks)]

    # Determine sequence length from the first sequence spec
    seq_len = history_length
    for spec in syn.sparse_features:
        if spec.is_sequence:
            seq_len = spec.sequence_length
            break

    return FeatureSchema(
        embedding_tables=tables,
        sequence_groups=sequence_groups,
        scalar_features=scalar_features,
        dense_features=dense,
        batch_to_feature={},
        sequence_length=seq_len,
        num_tasks=syn.num_tasks,
        task_names=task_names,
        embedding_dim=emb_dim,
        embedding_init=model_config.embedding_init,
    )
