from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""

    # Yambda dataset variant: "50m" (dev), "500m" (medium), "5b" (full scale)
    dataset_size: str = "50m"

    # Number of past events per behavior pool (listen+, like, skip).
    # Each pool is padded/truncated to this length.
    history_length: int = 20

    # How to select history events:
    #   "last_l" — most recent L events per pool
    #   "last_x_days" — all events within history_days window
    history_mode: str = "last_l"

    # Time window for "last_x_days" mode (ignored in "last_l" mode)
    history_days: int = 30

    # Deprecated: negative sampling removed in favor of natural event labels
    num_negatives: int = 0
    negative_sampler: str = "none"

    # Inactivity gap (seconds) that defines session boundaries.
    # Events separated by more than this gap belong to different sessions.
    session_gap_seconds: int = 1800  # 30 minutes

    # DataLoader worker processes (0 = main process only, required for CSAN)
    num_workers: int = 4

    # Enable user/item/cross counter features (log-transformed, multi-window)
    enable_counters: bool = False

    # Time windows (days) for counter aggregation, e.g. [7, 30] = 7-day + 30-day
    counter_windows_days: list[int] = field(default_factory=lambda: [30])

    # Root directory for raw and processed data
    data_dir: str = "data"

    # Cache preprocessed collated batches to disk for faster restarts
    use_cache: bool = True
    cache_dir: str = "data/cache"

    # Synthetic data generation (replaces real data when enabled)
    synthetic: SyntheticDataConfig = field(default_factory=lambda: SyntheticDataConfig())

    # Schema: a file path (str) or inline SchemaConfig.
    # When a string, the file is loaded and parsed during Config.load().
    schema: SchemaConfig | str = field(default_factory=lambda: SchemaConfig())


# ---------------------------------------------------------------------------
# Feature schema configuration (YAML-driven)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingTableConfig:
    """One embedding table: name, features, and optional size/dim/pooling."""
    name: str = ""
    features: list[str] = field(default_factory=list)
    num_embeddings: int = 0
    embedding_dim: int = 0
    pooling: str = "none"

# Tower routing constants (shared by ScalarFeatureSpec and DenseFeatureSpec)
TOWER_BOTH = "both"
TOWER_USER = "user"
TOWER_ITEM = "item"


@dataclass
class ScalarFeatureSpec:
    """One scalar (embedding) feature with tower routing.

    DLRM uses ``tower`` to route scalars to user or item towers.
    OneTrans ignores ``tower`` (all scalars go to the NS tokenizer).

    In YAML, scalar_features accepts both formats:
      scalar_features: [uid, item, artist, album]          # plain strings
      scalar_features:                                      # explicit specs
        - { name: uid, tower: user }
        - { name: item, tower: item }
    """
    name: str = ""
    tower: str = TOWER_BOTH

    def in_user_tower(self) -> bool:
        return self.tower in (TOWER_USER, TOWER_BOTH)

    def in_item_tower(self) -> bool:
        return self.tower in (TOWER_ITEM, TOWER_BOTH)

@dataclass
class DenseFeatureSpec:
    """One dense (non-embedding) feature.

    When ``project`` is True, the feature is projected to ``embedding_dim``
    via a learned Linear+GELU before concatenation into NS tokens.  When
    False, the raw values are concatenated directly (lower parameter count,
    but contributes ``dim`` instead of ``embedding_dim`` to the NS input).

    The ``tower`` field controls which DLRM tower receives this feature:
      "both" (default) — included in both user and item towers
      "user" — user tower only
      "item" — item tower only
    OneTrans ignores this field (all dense features go to the NS tokenizer).
    """
    name: str = ""
    dim: int = 0
    value_range_min: float = 0.0
    value_range_max: float = 1.0
    project: bool = True
    activation: str = "gelu"
    tower: str = TOWER_BOTH

    def in_user_tower(self) -> bool:
        return self.tower in (TOWER_USER, TOWER_BOTH)

    def in_item_tower(self) -> bool:
        return self.tower in (TOWER_ITEM, TOWER_BOTH)


@dataclass
class FeatureConfig:
    """Feature definitions: what features exist and how they are grouped."""
    sequence_groups: dict[str, list[str]] = field(default_factory=dict)
    scalar_features: list[ScalarFeatureSpec] = field(default_factory=list)
    dense_features: list[DenseFeatureSpec] = field(default_factory=list)

    def __post_init__(self):
        self.scalar_features = [
            ScalarFeatureSpec(name=s) if isinstance(s, str) else s
            for s in self.scalar_features
        ]

    @property
    def scalar_feature_names(self) -> list[str]:
        """All scalar feature names in config order."""
        return [sf.name for sf in self.scalar_features]

    @property
    def user_scalar_features(self) -> list[ScalarFeatureSpec]:
        """Scalar features routed to the user tower."""
        return [sf for sf in self.scalar_features if sf.in_user_tower()]

    @property
    def item_scalar_features(self) -> list[ScalarFeatureSpec]:
        """Scalar features routed to the item tower."""
        return [sf for sf in self.scalar_features if sf.in_item_tower()]

    def all_ec_feature_names(self) -> list[str]:
        """All EC feature names in deterministic order (scalars first)."""
        names = list(self.scalar_feature_names)
        for feats in self.sequence_groups.values():
            names.extend(feats)
        return names


@dataclass
class SchemaConfig:
    """Data-pipeline mapping loaded from a schema file or inline.

    Maps batch dict keys to EC feature names and defines KJT ordering.
    """
    batch_to_feature: dict[str, str] = field(default_factory=dict)
    kjt_feature_order: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic data configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataConfig:
    """Synthetic random data generation config.

    When ``enabled=True``, training uses randomly generated data instead of
    loading the real Yambda dataset.  Embedding table sizes, feature counts,
    value ranges, and dense feature dimensions are all configurable.  Feature
    names are auto-generated (``f_t0_0``, ``dense_0``, ``task0``, etc.).
    """
    enabled: bool = False
    seed: int = 42
    num_samples: int = 100_000
    num_prebatched: int = 16
    label_positive_rate: float = 0.3

    # Sparse ID value range: IDs are randint(sparse_id_min, sparse_id_max).
    # 0 for sparse_id_max means use each table's num_embeddings.
    sparse_id_min: int = 0
    sparse_id_max: int = 0

    # Sparse sequence length range: each feature gets randint(len_min, len_max+1) IDs.
    # 0 for both means use fixed length from schema.sequence_length.
    sparse_len_min: int = 0
    sparse_len_max: int = 0



@dataclass
class OneTransConfig:
    """OneTrans Transformer architecture hyperparameters."""

    # Hidden dimension throughout the Transformer stack
    d_model: int = 128

    # Number of attention heads (d_model must be divisible by n_heads)
    n_heads: int = 4

    # Number of stacked OneTrans blocks
    n_layers: int = 4

    # FFN hidden dim multiplier: ffn_dim = d_model * ffn_mult
    ffn_mult: int = 4

    # Number of non-sequential (NS) tokens from the auto-split tokenizer.
    # These represent candidate item + user context features.
    n_ns_tokens: int = 8

    # Progressive S-token query pruning across layers.
    # Reduces S-token queries linearly from L_S (layer 0) to L_NS (last layer).
    use_pyramid: bool = True

    # Dropout rate for attention and FFN
    dropout: float = 0.1

    # Learnable positional embeddings for S-tokens (one per position per pool)
    pos_embed: bool = True


@dataclass
class ModelConfig:
    """Model architecture selection and shared hyperparameters."""

    # Model architecture: "dlrm" (DLRM++ baseline) or "onetrans" (OneTrans Transformer)
    model_type: str = "dlrm"

    # Embedding dimension for all tables (default when per-table dim is 0)
    embedding_dim: int = 64

    # DLRM interaction module: "concat_mlp" | "dot" | "dcnv2"
    interaction_type: str = "concat_mlp"

    # MLP layer sizes for DLRM bottom (user/item tower) network
    bottom_mlp_dims: list[int] = field(default_factory=lambda: [64])

    # MLP layer sizes for DLRM top (after interaction) network
    top_mlp_dims: list[int] = field(default_factory=lambda: [128, 64])

    # Number of cross layers for DCNv2 interaction mode
    dcn_num_cross_layers: int = 3

    # Dropout rate for MLP layers
    dropout: float = 0.1

    # Embedding weight initialization:
    #   "uniform" — TorchRec default (uniform within bounds)
    #   "normal"  — nn.Embedding default (normal distribution, std=1)
    embedding_init: str = "uniform"

    # Embedding tables: table names, features, sizes, and per-table pooling.
    embedding_tables: list[EmbeddingTableConfig] = field(default_factory=list)

    # OneTrans-specific hyperparameters (only used when model_type="onetrans")
    onetrans: OneTransConfig = field(default_factory=OneTransConfig)

    def resolved_embedding_tables(self) -> list[EmbeddingTableConfig]:
        """Return embedding tables with embedding_dim resolved to model default."""
        D = self.embedding_dim
        return [
            EmbeddingTableConfig(
                name=t.name, features=t.features,
                num_embeddings=t.num_embeddings,
                embedding_dim=t.embedding_dim if t.embedding_dim > 0 else D,
                pooling=t.pooling,
            )
            for t in self.embedding_tables
        ]



@dataclass
class TrainConfig:
    """Training loop configuration."""

    # Global batch size (split across all GPUs in distributed training)
    batch_size: int = 256

    # Learning rate for dense (non-embedding) parameters
    lr: float = 1e-3

    # Separate learning rate for embedding parameters (via FBGEMM fused optimizer)
    embedding_lr: float = 1e-2

    # Embedding optimizer: "adam" or "row_wise_adagrad"
    embedding_optimizer: str = "adam"

    # Embedding optimizer epsilon
    embedding_eps: float = 1e-8

    # AdamW weight decay coefficient
    weight_decay: float = 1e-5

    # Number of training epochs
    epochs: int = 10

    # Dense warmup steps before cosine decay schedule
    warmup_steps: int = 1000

    # Dense warmup: initial LR multiplier (LR starts at lr * warmup_start_factor)
    warmup_start_factor: float = 0.01

    # Sparse (embedding) warmup: number of steps and initial LR multiplier.
    # During warmup, sparse LR = embedding_lr * sparse_warmup_value.
    # Set sparse_warmup_steps=0 to disable.
    sparse_warmup_steps: int = 0
    sparse_warmup_value: float = 0.1

    # Enable BF16 mixed precision (torch.amp.autocast).
    # Forward/backward matmuls run in BF16; weights, optimizer state,
    # reductions, and norms stay in FP32.
    bf16: bool = True

    # Enable TF32 for FP32 matmuls (hardware-level reduced precision).
    # Only affects FP32 ops not covered by BF16 autocast.
    # No effect when bf16=true (autocast overrides all matmuls to BF16).
    allow_tf32: bool = False

    # Max gradient norm for gradient clipping (dense params only;
    # embedding grads are handled by FBGEMM fused optimizer)
    grad_clip: float = 1.0

    # Shuffle training data each epoch
    shuffle: bool = True

    # Log training metrics every N steps
    log_interval: int = 100

    # Run evaluation every N epochs
    eval_interval: int = 1

    # Directory for checkpoints and results
    checkpoint_dir: str = "results"

    # Random seed for reproducibility
    seed: int = 42

    # Per-task loss weights.  Task names are defined by the YAML config
    # (e.g. "listen_plus", "like", "task0").  BCE loss is used by default;
    # tasks listed in regression_tasks use MSE instead.
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "task0": 1.0,
    })

    # Tasks that use MSE loss instead of BCE. All other tasks use BCE.
    regression_tasks: list[str] = field(default_factory=list)

    # Dense optimizer: "adamw" or "shampoo"
    # Shampoo is a 2nd-order optimizer that uses matrix preconditioning
    # for faster convergence. (https://github.com/facebookresearch/optimizers)
    dense_optimizer: str = "adamw"

    # Shampoo: how often to recompute preconditioner (eigendecomp).
    # Higher = less overhead but slower adaptation. Typical: 100-1000.
    shampoo_precondition_frequency: int = 1000

    # Shampoo: max dimension for preconditioner matrices.
    # Params with dim > this are blocked into smaller chunks.
    shampoo_max_preconditioner_dim: int = 4096

    # Shampoo: use BF16 for factor matrices in the preconditioner.
    # WARNING: Setting this to true with TF32 matmuls on MI350X triggers
    # a non-deterministic NaN in the eigenvalue decomposition.
    shampoo_use_bf16_factor_matrix: bool = False

    # Shampoo: momentum (0.0 = disabled). Reproducer uses 0.5.
    shampoo_momentum: float = 0.0

    # Shampoo: use Nesterov momentum
    shampoo_use_nesterov: bool = False

    # Shampoo: use decoupled weight decay (like AdamW)
    shampoo_use_decoupled_weight_decay: bool = False

    # Weight for in-batch BPR contrastive loss (0.0 = disabled)
    contrastive_weight: float = 0.0

    # Temperature for contrastive loss softmax
    contrastive_temperature: float = 0.1


@dataclass
class EmbeddingShardingConfig:
    """TorchRec DMP embedding sharding configuration."""

    # Enable embedding sharding across GPUs via DistributedModelParallel
    enabled: bool = False

    # Sharding strategy per table:
    #   "auto" — TorchRec EmbeddingShardingPlanner decides per-table
    #   "table_wise" — whole tables on individual GPUs
    #   "row_wise" — table rows split across GPUs
    #   "data_parallel" — replicate all tables (equivalent to DDP)
    strategy: str = "auto"


@dataclass
class DistributedConfig:
    """Multi-GPU / multi-node training configuration."""

    # Enable distributed training
    enabled: bool = False

    # Dense parameter replication strategy:
    #   "ddp" — DistributedDataParallel (allreduce gradients)
    #   "fsdp" — FullyShardedDataParallel (shard params + grads + optimizer)
    dense_strategy: str = "ddp"

    # Embedding sharding via TorchRec DMP (model-parallel for embedding tables)
    embedding_sharding: EmbeddingShardingConfig = field(
        default_factory=EmbeddingShardingConfig,
    )

    # Cluster topology
    num_nodes: int = 1
    gpus_per_node: int = 8


@dataclass
class Config:
    """Top-level configuration."""

    feature: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    @property
    def task_names(self) -> list[str]:
        """Active task names derived from train.loss_weights."""
        active = [k for k, v in self.train.loss_weights.items() if v > 0]
        return active if active else ["task0"]

    def feature_to_batch_key(self) -> dict[str, str]:
        """EC feature name → batch dict key mapping."""
        inv = {v: k for k, v in self.data.schema.batch_to_feature.items()}
        for name in self.feature.all_ec_feature_names():
            if name not in inv:
                inv[name] = name
        return inv

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = _from_dict(cls, raw)

        if isinstance(config.data.schema, str) and config.data.schema:
            schema_path = Path(config.data.schema)
            if not schema_path.is_absolute():
                schema_path = path.parent / schema_path
            with open(schema_path) as f:
                config.data.schema = _from_dict(SchemaConfig, yaml.safe_load(f) or {})

        return config


_DC_REGISTRY: dict[str, type] = {}


def _from_dict(dc_cls: type, raw: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a nested dict."""
    if raw is None:
        return dc_cls()
    if not _DC_REGISTRY:
        for cls in (
            DataConfig, ModelConfig, TrainConfig, OneTransConfig,
            DistributedConfig, EmbeddingShardingConfig,
            FeatureConfig, SchemaConfig, EmbeddingTableConfig,
            SyntheticDataConfig,
            DenseFeatureSpec, ScalarFeatureSpec,
        ):
            _DC_REGISTRY[cls.__name__] = cls
    kwargs = {}
    for f in fields(dc_cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if hasattr(f.type, "__dataclass_fields__") or type_name in _DC_REGISTRY:
            sub_cls = _DC_REGISTRY.get(type_name)
            if sub_cls and isinstance(val, dict):
                val = _from_dict(sub_cls, val)
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            # Handle list[SomeDataclass] fields by matching the type hint
            inner_cls = _guess_list_inner_dc(type_name)
            if inner_cls is not None:
                val = [_from_dict(inner_cls, v) if isinstance(v, dict) else v for v in val]
        kwargs[f.name] = val
    return dc_cls(**kwargs)


def _guess_list_inner_dc(type_hint: str) -> type | None:
    """Extract the inner dataclass from a 'list[Foo]' type hint string."""
    import re
    m = re.search(r"list\[(\w+)\]", type_hint, re.IGNORECASE)
    if m:
        return _DC_REGISTRY.get(m.group(1))
    return None
