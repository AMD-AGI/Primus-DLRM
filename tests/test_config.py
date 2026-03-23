"""Smoke tests for config loading/saving."""
import tempfile
from pathlib import Path

from primus_dlrm.config import Config


def test_default_config():
    cfg = Config()
    assert cfg.data.dataset_size == "50m"
    assert cfg.model.embedding_dim == 16
    assert cfg.train.batch_size == 256


def test_roundtrip():
    cfg = Config()
    cfg.model.embedding_dim = 128
    cfg.train.epochs = 5

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        cfg.save(f.name)
        loaded = Config.load(f.name)

    assert loaded.model.embedding_dim == 128
    assert loaded.train.epochs == 5
    assert loaded.data.dataset_size == "50m"


def test_load_baseline_config():
    cfg = Config.load("configs/dlrm_baseline.yaml")
    assert cfg.data.history_length == 100
    assert cfg.model.interaction_type == "concat_mlp"
    assert cfg.train.bf16 is True
