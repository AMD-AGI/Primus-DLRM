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

    # How many raw events to scan backward when building per-behavior pools.
    # The dataset reads the last ``scan_window`` events of the user's history
    # and splits them into listen+/like/skip; ``history_length`` then keeps
    # the last L of each. Default 500 matches legacy behavior; raising this
    # gives the per-pool ``last_l`` slice access to a much larger pool of
    # candidates (important when listen+/like/skip frequencies are imbalanced).
    # Higher values increase per-sample data-loading cost roughly linearly.
    scan_window: int = 500

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
    # Number of batches each worker pre-loads ahead of consumption
    prefetch_factor: int = 2

    # Enable user/item/cross counter features (log-transformed, multi-window)
    enable_counters: bool = False

    # Time windows (days) for counter aggregation, e.g. [7, 30] = 7-day + 30-day
    counter_windows_days: list[int] = field(default_factory=lambda: [30])

    # Root directory for raw and processed data. All other paths
    # (processed/, cache/, shared_metadata/) are derived from this and
    # ``dataset_size`` via ``primus_dlrm.data.dataset.DataPaths``.
    data_dir: str = "data"

    # Cache preprocessed collated batches to disk for faster restarts.
    use_cache: bool = True

    # Synthetic data generation (replaces real data when enabled)
    synthetic: SyntheticDataConfig = field(default_factory=lambda: SyntheticDataConfig())

    # Schema: a file path (str) or inline SchemaConfig.
    # When a string, the file is loaded and parsed during Config.load().
    schema: SchemaConfig | str = field(default_factory=lambda: SchemaConfig())


# ---------------------------------------------------------------------------
# Feature schema configuration (YAML-driven)
# ---------------------------------------------------------------------------

@dataclass
class TableShardingSpec:
    """Per-table TorchRec planner constraints.

    All fields are optional; any unset field falls back to the global
    ``distributed.embedding_sharding.strategy`` and the planner's auto choice.
    Maps directly onto ``torchrec.distributed.planner.types.ParameterConstraints``:

    - ``sharding_type``: one of ``"row_wise" | "table_wise" | "column_wise" |
      "data_parallel"`` (passed as a singleton ``sharding_types`` list).
    - ``compute_kernel``: one of ``"fused" | "fused_uvm" | "fused_uvm_caching"
      | "dense"`` (singleton ``compute_kernels`` list). ``fused_uvm_caching``
      requires ``cache_load_factor``.
    - ``cache_load_factor``: forwarded into ``CacheParams.load_factor`` when
      the compute kernel uses caching; ignored otherwise.
    - ``min_partition``: minimum row count per shard (RW/CW); useful to keep
      small tables from being split below GPU-efficient grain.
    - ``enforce_hbm``: when True, planner refuses to spill this table to UVM
      even under HBM pressure.
    - ``output_dtype``: ``"fp32" | "fp16" | "bf16"`` for the embedding output
      tensor (the table itself stays in its weight precision).
    - ``ranks``: pin a TABLE_WISE table to specific rank(s) via TorchRec's
      ``device_group``. Honored by TW; advisory for other sharding types.
    """
    sharding_type: str | None = None
    compute_kernel: str | None = None
    cache_load_factor: float | None = None
    min_partition: int | None = None
    enforce_hbm: bool | None = None
    output_dtype: str | None = None
    ranks: list[int] | None = None


@dataclass
class EmbeddingTableConfig:
    """One embedding table: name, features, and optional size/dim/pooling."""
    name: str = ""
    features: list[str] = field(default_factory=list)
    num_embeddings: int = 0
    embedding_dim: int = 0
    pooling: str = "none"

    # Optional per-table TorchRec planner constraints. Inline here so authors
    # see the table size and its sharding choice in one place. ``None`` means
    # "fall back to the global default" (see ``EmbeddingShardingConfig``).
    sharding: TableShardingSpec | None = None


# Recognized cross-feature key sources resolvable from a single training event.
# Anchored on the ScoringPair fields plus a derived hour-of-day from the event
# timestamp.  Extending this set requires a matching getter in
# ``YambdaTrainDataset.__getitem__`` and ``YambdaEvalDataset.get_user_history``.
CROSS_FEATURE_KEYS = ("uid", "item_id", "artist_id", "album_id", "hour_of_day", "is_organic")


@dataclass
class CrossFeatureSpec:
    """One cross-product embedding table, optionally hashed.

    Each spec auto-registers an ``EmbeddingCollection`` table at config-load
    time via ``Config.expand_cross_features()``. The dataset hot path computes
    ``xxhash64(keys) % num_embeddings`` per sample (see
    ``primus_dlrm.data.hashing``) and feeds the result as the EC index.

    Hashed vs unique coverage:
      - ``num_embeddings >= prod(unique_cardinality(k) for k in keys)``:
        the cross is "unique-covered" (e.g. ``user_x_hour`` with 1M users x 24
        hours = 24M unique pairs at num_embeddings=24M). Hash collisions only
        come from xxhash birthday probability, effectively zero.
      - ``num_embeddings <  prod(...)``: the cross is "hashed-truncated" (e.g.
        ``user_x_artist`` with 1M x 1.29M = 1.29T unique pairs squeezed into
        100M buckets). Multiple unique pairs share buckets; the table operates
        as a memorization shortcut rather than a unique row per pair.

    Same name ``num_embeddings`` as ``EmbeddingTableConfig.num_embeddings``;
    the auto-registered table inherits this value 1:1, so the hash modulus
    always matches the storage row count by construction (no out-of-range
    indices possible).

    YAML example::

        data:
          cross_features:
            - { name: user_x_artist, keys: [uid, artist_id],   num_embeddings: 100000000 }  # hashed (1.29T unique > 100M)
            - { name: user_x_hour,   keys: [uid, hour_of_day], num_embeddings: 24000000  }  # unique (1M x 24)
            - { name: user_x_album,  keys: [uid, album_id],    num_embeddings: 40000000, enabled: false }

    ``enabled: false`` disables a single cross without removing the spec from
    the YAML, useful for ablation sweeps.
    """
    name: str = ""
    keys: list[str] = field(default_factory=list)
    num_embeddings: int = 0
    salt: int = 0
    enabled: bool = True

    # Optional per-table TorchRec planner constraints. Propagated into the
    # auto-registered ``EmbeddingTableConfig`` by ``Config.expand_cross_features``.
    sharding: TableShardingSpec | None = None

# Tower routing constants (DLRM only; OneTrans ignores tower)
TOWER_USER = "user"
TOWER_ITEM = "item"


class TowerRoutingMixin:
    """Shared tower routing for DLRM feature specs.

    The ``tower`` field controls which DLRM tower receives this feature:
      "" (default) — included in both towers
      "user" — user tower only
      "item" — item tower only
    OneTrans ignores this field.
    """
    tower: str = ""

    def in_user_tower(self) -> bool:
        return self.tower != TOWER_ITEM

    def in_item_tower(self) -> bool:
        return self.tower != TOWER_USER


@dataclass
class ScalarFeatureSpec(TowerRoutingMixin):
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


@dataclass
class DenseFeatureSpec(TowerRoutingMixin):
    """One dense (non-embedding) feature.

    When ``project`` is True, the feature is projected to ``embedding_dim``
    via a learned Linear+GELU before concatenation into NS tokens.  When
    False, the raw values are concatenated directly (lower parameter count,
    but contributes ``dim`` instead of ``embedding_dim`` to the NS input).
    """
    name: str = ""
    dim: int = 0
    value_range_min: float = 0.0
    value_range_max: float = 1.0
    project: bool = True
    activation: str = "gelu"


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
class TransformerConfig:
    """Transformer architecture hyperparameters."""

    # Hidden dimension throughout the Transformer stack
    d_model: int = 128

    # Number of attention heads (d_model must be divisible by n_heads)
    n_heads: int = 4

    # Number of stacked Transformer blocks
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

    # Attention implementation:
    #   "sdpa"  — PyTorch scaled_dot_product_attention (default, works everywhere)
    #   "fav2"  — FlashAttention-2 (requires flash_attn package)
    #   "fav4"  — FlashAttention-4 / CuTeDSL (requires flash-attn-4, Blackwell optimized)
    #   "turbo" — Primus-Turbo flash attention (requires primus_turbo package, ROCm)
    attention_impl: str = "turbo"

    # Jagged attention: when True, build cu_seqlens from per-pool real lengths
    # (data.scan_window provides the events; data.history_length caps each pool)
    # and call flash_attn_varlen_func instead of dense flash_attn_func.
    # Skips compute on padded zero tokens. Requires attention_impl='fav2_varlen'
    # or compatible varlen-capable backend; ignored otherwise.
    # Pyramid masking (use_pyramid=True) is NOT supported with jagged in this
    # MVP — set use_pyramid=False when enabling use_jagged.
    use_jagged: bool = False

    # When True, recompute each transformer block in backward instead of
    # retaining its activations. On the packed (use_jagged=True) path this
    # frees the per-layer interleave buffer (attn_qkv) and FA-output buffers,
    # cutting peak HBM by ~60% (e.g. 250 GB -> 96 GB at hist=500) and
    # unlocking history_length up to 2000 on a single MI355X. Same weights
    # and same outputs; ~25% extra step time from recompute.
    grad_checkpoint: bool = False



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

    # Cross-product (optionally hashed) embedding tables (e.g. user x artist,
    # user x hour, user x artist x hour). Sits next to embedding_tables because
    # each enabled spec auto-registers as one EmbeddingCollection table at
    # load time via Config.expand_cross_features(), inheriting embedding_dim
    # and being routed through the same DMP planner. The dataset reads the
    # same list to compute xxhash64 ids per sample. Toggle individual crosses
    # on/off via the per-spec ``enabled`` flag.
    cross_features: list[CrossFeatureSpec] = field(default_factory=list)

    # Transformer hyperparameters (used when model_type="onetrans")
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    def resolved_embedding_tables(self) -> list[EmbeddingTableConfig]:
        """Return embedding tables with embedding_dim resolved to model default."""
        D = self.embedding_dim
        return [
            EmbeddingTableConfig(
                name=t.name, features=t.features,
                num_embeddings=t.num_embeddings,
                embedding_dim=t.embedding_dim if t.embedding_dim > 0 else D,
                pooling=t.pooling,
                sharding=t.sharding,
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

    # Log maximum GPU memory over all ranks via all_gather at each log_interval.
    # Adds a collective sync, disable for perf runs.
    log_max_gpu_memory: bool = False

    # Run evaluation every N epochs
    eval_interval: int = 1

    # Directory for checkpoints and results
    checkpoint_dir: str = "results"
    save_checkpoint: bool = True

    # Whether to save checkpoints at end of each epoch
    save_checkpoint: bool = False

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

    # Shampoo: use upstream DDPDistributedConfig, which shards Shampoo's
    # optimizer-state computation and synchronizes updates across local ranks.
    shampoo_use_ddp_distributed_config: bool = False

    # Shampoo DDPDistributedConfig: number of trainers per distributed group.
    # -1 lets upstream default to LOCAL_WORLD_SIZE.
    shampoo_ddp_num_trainers_per_group: int = -1

    # Shampoo DDPDistributedConfig: communicate updated parameters instead of
    # parameter updates. Default mirrors upstream DDPDistributedConfig.
    shampoo_ddp_communicate_params: bool = False

    # torch.compile: compile each transformer block with Inductor (or other backend)
    torch_compile: bool = False
    # Backend for torch.compile: "inductor" (default) or "aot_eager".
    # NOTE: inductor crashes with attention_impl="sdpa" on ROCm due to a
    # workspace sizing bug in SDPA's backward pass.  Use "fav2" or "turbo"
    # attention when compiling with inductor.
    torch_compile_backend: str = "inductor"

    # Weight for in-batch BPR contrastive loss (0.0 = disabled)
    contrastive_weight: float = 0.0

    # Temperature for contrastive loss softmax
    contrastive_temperature: float = 0.1


@dataclass
class TopologyConfig:
    """Per-rank resource hints fed to the TorchRec planner's ``Topology``.

    All fields are optional; unset means "use TorchRec's default". The
    environment variables ``PRIMUS_TORCHREC_HBM_CAP_GB``,
    ``PRIMUS_TORCHREC_DDR_CAP_GB`` and ``PRIMUS_TORCHREC_LOCAL_WORLD_SIZE``
    take precedence over these YAML values, so operators can do one-off
    overrides without editing the config.

    Recommended starting values per platform (~90% of physical HBM, leaving
    headroom for activations + optimizer state):

    - MI355X (288 GB physical):  ``hbm_cap_gb: 260``
    - B200   (178 GB physical):  ``hbm_cap_gb: 161``
    - H100   ( 80 GB physical):  ``hbm_cap_gb:  72``
    """
    # Per-rank HBM cap in GB exposed to the planner.
    hbm_cap_gb: float | None = None

    # Per-rank DDR cap in GB; controls how much can be spilled to UVM.
    ddr_cap_gb: float | None = None

    # GPUs per host (TorchRec uses this for local-vs-remote bandwidth modelling).
    local_world_size: int | None = None


@dataclass
class EmbeddingShardingConfig:
    """TorchRec DMP embedding sharding configuration."""

    # Enable embedding sharding across GPUs via DistributedModelParallel
    enabled: bool = False

    # Global default sharding strategy per table when the table itself does
    # not provide an inline ``sharding.sharding_type`` override:
    #   "auto" — TorchRec EmbeddingShardingPlanner decides per-table
    #   "table_wise" — whole tables on individual GPUs
    #   "row_wise" — table rows split across GPUs
    #   "data_parallel" — replicate all tables (equivalent to DDP)
    strategy: str = "auto"

    # Default per-table sharding spec applied to tables without an inline
    # ``sharding`` block. Per-table inline ``sharding`` wins field-by-field.
    default_table_sharding: TableShardingSpec | None = None

    # Resource hints fed to the planner's Topology (per-rank HBM/DDR caps,
    # local_world_size). Env vars override the YAML values.
    topology: TopologyConfig = field(default_factory=TopologyConfig)


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

    def expand_cross_features(self) -> list[CrossFeatureSpec]:
        """Auto-register enabled cross_features into the downstream lists.

        Mutates in place:
          - ``model.embedding_tables`` += one ``EmbeddingTableConfig`` per
            enabled spec (``embedding_dim=0`` so it inherits ``model.embedding_dim``).
          - ``feature.scalar_features`` += one ``ScalarFeatureSpec`` per spec
            (NS-token concat order = cross-spec order).
          - ``data.schema.batch_to_feature`` += ``{f"{spec.name}_id": spec.name}``.
          - ``data.schema.kjt_feature_order`` += ``spec.name`` (only when the
            schema's existing kjt_feature_order is non-empty -- otherwise the
            schema falls back to ``feature.all_ec_feature_names()`` which
            already picks up the new scalar feature).

        Specs with ``enabled=False`` are skipped entirely (no table, no scalar
        feature, no schema entry, no dataset hash).

        Idempotent: re-running is a no-op once expansion has happened. The
        first call records auto-registered names on ``self._cross_registered``
        so subsequent calls can distinguish "already auto-registered" from
        "user wrote a native table that collides with a cross spec name".

        Raises ``ValueError`` on:
          - a spec name colliding with a hand-written native table or scalar;
          - an unknown key (must be one of ``CROSS_FEATURE_KEYS``);
          - missing required fields (``name``, ``num_embeddings``, 2+ keys).

        Returns the list of specs that were actually expanded on this call.
        """
        if isinstance(self.data.schema, str):
            return []  # schema not yet loaded; expand_cross_features() runs after Config.load()

        registered: set[str] = getattr(self, "_cross_registered", set())
        existing_table_names = {t.name for t in self.model.embedding_tables}
        existing_scalar_names = set(self.feature.scalar_feature_names)

        expanded: list[CrossFeatureSpec] = []
        for spec in self.model.cross_features:
            if not spec.enabled:
                continue
            if not spec.name:
                raise ValueError("CrossFeatureSpec.name is required")
            if spec.num_embeddings <= 0:
                raise ValueError(f"CrossFeatureSpec '{spec.name}' needs num_embeddings > 0")
            if not spec.keys or len(spec.keys) < 2:
                raise ValueError(
                    f"CrossFeatureSpec '{spec.name}' needs at least 2 keys, got {spec.keys}"
                )
            for k in spec.keys:
                if k not in CROSS_FEATURE_KEYS:
                    raise ValueError(
                        f"CrossFeatureSpec '{spec.name}' uses unknown key '{k}'; "
                        f"recognized keys: {CROSS_FEATURE_KEYS}"
                    )

            if spec.name in registered:
                continue  # idempotent: already auto-registered by an earlier call

            if spec.name in existing_table_names:
                raise ValueError(
                    f"CrossFeatureSpec '{spec.name}' collides with an existing "
                    f"native embedding table; rename the cross spec or the native table"
                )
            if spec.name in existing_scalar_names:
                raise ValueError(
                    f"CrossFeatureSpec '{spec.name}' collides with an existing "
                    f"scalar feature; rename the cross spec or the scalar feature"
                )

            self.model.embedding_tables.append(
                EmbeddingTableConfig(
                    name=spec.name,
                    features=[spec.name],
                    num_embeddings=spec.num_embeddings,
                    embedding_dim=0,
                    pooling="none",
                    sharding=spec.sharding,
                )
            )
            self.feature.scalar_features.append(ScalarFeatureSpec(name=spec.name))

            batch_key = f"{spec.name}_id"
            if batch_key not in self.data.schema.batch_to_feature:
                self.data.schema.batch_to_feature[batch_key] = spec.name
            if (
                self.data.schema.kjt_feature_order
                and spec.name not in self.data.schema.kjt_feature_order
            ):
                self.data.schema.kjt_feature_order.append(spec.name)

            existing_table_names.add(spec.name)
            existing_scalar_names.add(spec.name)
            registered.add(spec.name)
            expanded.append(spec)

        self._cross_registered = registered
        return expanded

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

        # cross_features moved from data.cross_features to model.cross_features.
        # Catch stale YAMLs at load time so the operator sees a clear error
        # instead of training with zero crosses (the old field would silently
        # be ignored by _from_dict since DataConfig no longer declares it).
        if isinstance(raw, dict) and "cross_features" in (raw.get("data") or {}):
            raise ValueError(
                f"{path}: 'cross_features' moved from 'data.cross_features' to "
                f"'model.cross_features'; please move the block under model:."
            )

        config = _from_dict(cls, raw)

        if isinstance(config.data.schema, str) and config.data.schema:
            schema_path = Path(config.data.schema)
            if not schema_path.is_absolute():
                schema_path = path.parent / schema_path
            with open(schema_path) as f:
                config.data.schema = _from_dict(SchemaConfig, yaml.safe_load(f) or {})

        # Auto-register cross-feature specs into embedding_tables / scalar_features
        # / schema lists.  Runs after the schema is materialized so kjt_feature_order
        # mutations land on the right object.
        config.expand_cross_features()

        return config


_DC_REGISTRY: dict[str, type] = {}


def _from_dict(dc_cls: type, raw: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a nested dict."""
    if raw is None:
        return dc_cls()
    if not _DC_REGISTRY:
        for cls in (
            DataConfig, ModelConfig, TrainConfig, TransformerConfig,
            DistributedConfig, EmbeddingShardingConfig, TopologyConfig,
            FeatureConfig, SchemaConfig, EmbeddingTableConfig,
            TableShardingSpec,
            SyntheticDataConfig,
            DenseFeatureSpec, ScalarFeatureSpec,
            CrossFeatureSpec,
        ):
            _DC_REGISTRY[cls.__name__] = cls
    kwargs = {}
    for f in fields(dc_cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        # Strip ``| None`` / ``Optional[...]`` wrappers so Optional dataclass
        # fields still hit the registry-based recursive instantiation below.
        type_name = _strip_optional(type_name)
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


def _strip_optional(type_name: str) -> str:
    """Reduce ``"Foo | None"`` / ``"None | Foo"`` / ``"Optional[Foo]"`` to ``"Foo"``."""
    import re
    s = type_name.strip()
    m = re.fullmatch(r"Optional\[(.+)\]", s)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in s.split("|")]
    non_none = [p for p in parts if p and p != "None"]
    return non_none[0] if len(non_none) == 1 else s


def _guess_list_inner_dc(type_hint: str) -> type | None:
    """Extract the inner dataclass from a 'list[Foo]' type hint string."""
    import re
    m = re.search(r"list\[(\w+)\]", type_hint, re.IGNORECASE)
    if m:
        return _DC_REGISTRY.get(m.group(1))
    return None
