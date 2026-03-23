from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_size: str = "50m"
    history_length: int = 20
    history_mode: str = "last_l"  # "last_l" | "last_x_days"
    history_days: int = 30
    num_negatives: int = 0  # deprecated: negative sampling removed in favor of natural events
    negative_sampler: str = "none"  # deprecated
    session_gap_seconds: int = 1800  # 30 minutes
    num_workers: int = 4
    enable_counters: bool = False
    counter_windows_days: list[int] = field(default_factory=lambda: [30])
    data_dir: str = "data"
    use_cache: bool = True
    cache_dir: str = "data/cache"


@dataclass
class OneTransConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    n_ns_tokens: int = 8
    use_pyramid: bool = True
    dropout: float = 0.1
    pos_embed: bool = True


@dataclass
class ModelConfig:
    model_type: str = "dlrm"  # "dlrm" | "onetrans"
    embedding_dim: int = 16
    embedding_dim_small: int = 8
    interaction_type: str = "concat_mlp"  # "concat_mlp" | "dot" | "dcnv2"
    bottom_mlp_dims: list[int] = field(default_factory=lambda: [64])
    top_mlp_dims: list[int] = field(default_factory=lambda: [128, 64])
    dcn_num_cross_layers: int = 3
    dropout: float = 0.1
    audio_embed_dim: int = 256
    embedding_init: str = "uniform"  # "uniform" (TorchRec default) | "normal" (nn.Embedding default)
    onetrans: OneTransConfig = field(default_factory=OneTransConfig)


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    embedding_lr: float = 1e-2
    weight_decay: float = 1e-5
    epochs: int = 10
    warmup_steps: int = 1000
    bf16: bool = True
    grad_clip: float = 1.0
    shuffle: bool = True
    log_interval: int = 100
    eval_interval: int = 1
    checkpoint_dir: str = "results"
    seed: int = 42
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "listen_plus": 1.0,
    })
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1


@dataclass
class EmbeddingShardingConfig:
    enabled: bool = False
    strategy: str = "auto"  # "auto" | "table_wise" | "row_wise" | "data_parallel"


@dataclass
class DistributedConfig:
    enabled: bool = False
    dense_strategy: str = "ddp"  # "ddp" | "fsdp"
    embedding_sharding: EmbeddingShardingConfig = field(
        default_factory=EmbeddingShardingConfig,
    )
    num_nodes: int = 1
    gpus_per_node: int = 8


@dataclass
class Config:
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
