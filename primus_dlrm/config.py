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

    # Embedding dimension for main tables (item, artist, album, uid)
    embedding_dim: int = 16

    # Embedding dimension for small tables (event_type, is_organic, time_gap)
    embedding_dim_small: int = 8

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

    # Input dimension of audio embeddings from the dataset
    audio_embed_dim: int = 256

    # Embedding weight initialization:
    #   "uniform" — TorchRec default (uniform within bounds)
    #   "normal"  — nn.Embedding default (normal distribution, std=1)
    embedding_init: str = "uniform"

    # OneTrans-specific hyperparameters (only used when model_type="onetrans")
    onetrans: OneTransConfig = field(default_factory=OneTransConfig)


@dataclass
class TrainConfig:
    """Training loop configuration."""

    # Global batch size (split across all GPUs in distributed training)
    batch_size: int = 256

    # Learning rate for dense (non-embedding) parameters
    lr: float = 1e-3

    # Separate learning rate for embedding parameters (via FBGEMM fused optimizer)
    embedding_lr: float = 1e-2

    # AdamW weight decay coefficient
    weight_decay: float = 1e-5

    # Number of training epochs
    epochs: int = 10

    # Linear warmup steps before cosine decay schedule
    warmup_steps: int = 1000

    # Enable BF16 mixed precision (torch.amp.autocast).
    # Forward/backward matmuls run in BF16; weights, optimizer state,
    # reductions, and norms stay in FP32.
    bf16: bool = True

    # Enable TF32 for FP32 matmuls (hardware-level reduced precision).
    # Only affects FP32 ops not covered by BF16 autocast.
    # No effect when bf16=true (autocast overrides all matmuls to BF16).
    allow_tf32: bool = False

    # Enable FBGEMM TBE V2 embedding kernels.
    # Improves throughput ~14% by better overlapping embedding
    # computation with NCCL communication in the DMP pipeline.
    tbe_v2: bool = False

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

    # Per-task loss weights. Supported tasks:
    #   "listen_plus" (BCE) — primary: listened >= 50% played_ratio
    #   "like" (BCE) — explicit like event
    #   "dislike" (BCE) — explicit dislike event
    #   "played_ratio" (MSE) — continuous playback regression
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "listen_plus": 1.0,
    })

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
    """Top-level configuration combining data, model, training, and distributed settings."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return _from_dict(cls, raw)


def _from_dict(dc_cls: type, raw: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a nested dict."""
    if raw is None:
        return dc_cls()
    kwargs = {}
    for f in fields(dc_cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        _dc_registry = {
            "DataConfig": DataConfig, "ModelConfig": ModelConfig,
            "TrainConfig": TrainConfig, "OneTransConfig": OneTransConfig,
            "DistributedConfig": DistributedConfig,
            "EmbeddingShardingConfig": EmbeddingShardingConfig,
        }
        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if hasattr(f.type, "__dataclass_fields__") or type_name in _dc_registry:
            sub_cls = _dc_registry.get(type_name)
            if sub_cls and isinstance(val, dict):
                val = _from_dict(sub_cls, val)
        kwargs[f.name] = val
    return dc_cls(**kwargs)
