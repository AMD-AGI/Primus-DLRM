"""Smoke tests for model forward/backward."""
import torch
import pytest

from primus_dlrm.config import Config, DenseFeatureSpec, FeatureConfig, ModelConfig, TransformerConfig, SchemaConfig, EmbeddingTableConfig
from primus_dlrm.models.dlrm import DLRMBaseline
from primus_dlrm.models.onetrans import OneTransModel, pyramid_schedule
from primus_dlrm.training.losses import MultiTaskLoss


_YAMBDA_EMBEDDING_TABLES = [
    EmbeddingTableConfig("item", ["item", "hist_lp_item", "hist_like_item", "hist_skip_item"], num_embeddings=100),
    EmbeddingTableConfig("artist", ["artist", "hist_lp_artist", "hist_like_artist", "hist_skip_artist"], num_embeddings=50),
    EmbeddingTableConfig("album", ["album", "hist_lp_album", "hist_like_album", "hist_skip_album"], num_embeddings=30),
    EmbeddingTableConfig("uid", ["uid"], num_embeddings=200),
]

_DLRM_EMBEDDING_TABLES = [
    EmbeddingTableConfig("hist_item", ["hist_lp_item", "hist_like_item", "hist_skip_item"], num_embeddings=100, pooling="mean"),
    EmbeddingTableConfig("item", ["item"], num_embeddings=100),
    EmbeddingTableConfig("hist_artist", ["hist_lp_artist", "hist_like_artist", "hist_skip_artist"], num_embeddings=50, pooling="mean"),
    EmbeddingTableConfig("artist", ["artist"], num_embeddings=50),
    EmbeddingTableConfig("hist_album", ["hist_lp_album", "hist_like_album", "hist_skip_album"], num_embeddings=30, pooling="mean"),
    EmbeddingTableConfig("album", ["album"], num_embeddings=30),
    EmbeddingTableConfig("uid", ["uid"], num_embeddings=200),
]


_YAMBDA_SCHEMA = SchemaConfig(
    batch_to_feature={
        "item_id": "item", "artist_id": "artist", "album_id": "album",
        "hist_lp_item_ids": "hist_lp_item", "hist_lp_artist_ids": "hist_lp_artist",
        "hist_lp_album_ids": "hist_lp_album", "hist_like_item_ids": "hist_like_item",
        "hist_like_artist_ids": "hist_like_artist", "hist_like_album_ids": "hist_like_album",
        "hist_skip_item_ids": "hist_skip_item", "hist_skip_artist_ids": "hist_skip_artist",
        "hist_skip_album_ids": "hist_skip_album",
    },
    kjt_feature_order=[
        "item", "artist", "album",
        "hist_lp_item", "hist_like_item", "hist_skip_item",
        "hist_lp_artist", "hist_like_artist", "hist_skip_artist",
        "hist_lp_album", "hist_like_album", "hist_skip_album",
        "uid",
    ],
)


def _yambda_feature_config(audio_dim=32, enable_counters=False, counter_windows=None):
    """Build a FeatureConfig matching the Yambda layout for tests."""
    dense = [DenseFeatureSpec("audio_embed", audio_dim, project=True, activation="gelu")]
    if enable_counters and counter_windows:
        W = len(counter_windows)
        dense.append(DenseFeatureSpec("user_counters", 3 * W, project=False))
        dense.append(DenseFeatureSpec("item_counters", 3 * W, project=False))
        dense.append(DenseFeatureSpec("cross_counters", 9 * W, project=True, activation="relu"))
    fc = FeatureConfig(
        sequence_groups={
            "hist_lp": ["hist_lp_item", "hist_lp_artist", "hist_lp_album"],
            "hist_like": ["hist_like_item", "hist_like_artist", "hist_like_album"],
            "hist_skip": ["hist_skip_item", "hist_skip_artist", "hist_skip_album"],
        },
        scalar_features=["uid", "item", "artist", "album"],
        dense_features=dense,
    )
    tower_map = {"uid": "user", "item": "item", "artist": "item", "album": "item"}
    for sf in fc.scalar_features:
        sf.tower = tower_map.get(sf.name, "")
    return fc


def _make_dummy_batch(batch_size: int = 4, hist_len: int = 10, device: str = "cpu",
                      num_counter_windows: int = 0):
    batch = {
        "hist_lp_item_ids": torch.randint(0, 100, (batch_size, hist_len), device=device),
        "hist_lp_artist_ids": torch.randint(0, 50, (batch_size, hist_len), device=device),
        "hist_lp_album_ids": torch.randint(0, 30, (batch_size, hist_len), device=device),
        "hist_like_item_ids": torch.randint(0, 100, (batch_size, hist_len), device=device),
        "hist_like_artist_ids": torch.randint(0, 50, (batch_size, hist_len), device=device),
        "hist_like_album_ids": torch.randint(0, 30, (batch_size, hist_len), device=device),
        "hist_skip_item_ids": torch.randint(0, 100, (batch_size, hist_len), device=device),
        "hist_skip_artist_ids": torch.randint(0, 50, (batch_size, hist_len), device=device),
        "hist_skip_album_ids": torch.randint(0, 30, (batch_size, hist_len), device=device),
        "uid": torch.randint(0, 200, (batch_size,), device=device),
        "item_id": torch.randint(0, 100, (batch_size,), device=device),
        "artist_id": torch.randint(0, 50, (batch_size,), device=device),
        "album_id": torch.randint(0, 30, (batch_size,), device=device),
        "audio_embed": torch.randn(batch_size, 32, device=device),
        "listen_plus": torch.randint(0, 2, (batch_size,), device=device).float(),
        "like": torch.randint(0, 2, (batch_size,), device=device).float(),
        "dislike": torch.randint(0, 2, (batch_size,), device=device).float(),
        "listen_pct": torch.rand(batch_size, device=device),
    }
    if num_counter_windows > 0:
        W = num_counter_windows
        batch["user_counters"] = torch.rand(batch_size, 3 * W, device=device)
        batch["item_counters"] = torch.rand(batch_size, 3 * W, device=device)
        batch["cross_counters"] = torch.rand(batch_size, 9 * W, device=device)
    return batch


def _build_dlrm(model_config=None, device="cpu", tasks=None, num_counter_windows=0):
    """Build a DLRMBaseline from a full Config, matching the Yambda layout."""
    if tasks is None:
        tasks = ["listen_plus"]
    model_config = model_config or ModelConfig(
        embedding_dim=16, top_mlp_dims=[32, 16], bottom_mlp_dims=[32],
    )
    model_config.embedding_tables = list(_DLRM_EMBEDDING_TABLES)
    counter_days = list(range(1, num_counter_windows + 1)) if num_counter_windows > 0 else []
    full_config = Config()
    full_config.model = model_config
    full_config.feature = _yambda_feature_config(
        audio_dim=32,
        enable_counters=num_counter_windows > 0,
        counter_windows=counter_days if num_counter_windows > 0 else None,
    )
    full_config.data.schema = _YAMBDA_SCHEMA
    full_config.train.loss_weights = {t: 1.0 for t in tasks}
    return DLRMBaseline(full_config, device=torch.device(device))


def test_forward_cpu():
    model = _build_dlrm()
    batch = _make_dummy_batch()
    out = model(batch)

    assert "listen_plus" in out
    assert out["listen_plus"].shape == (4,)


def test_forward_multi_task():
    tasks = ["listen_plus", "like", "dislike", "listen_pct"]
    model = _build_dlrm(tasks=tasks)

    batch = _make_dummy_batch()
    out = model(batch)

    for t in tasks:
        assert t in out
        assert out[t].shape == (4,)


def test_backward_cpu():
    model = _build_dlrm()

    batch = _make_dummy_batch()
    out = model(batch)

    loss_fn = MultiTaskLoss(weights={"listen_plus": 1.0})
    labels = {"listen_plus": batch["listen_plus"]}
    total_loss, task_losses = loss_fn(out, labels)

    total_loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0
    assert not torch.isnan(total_loss)


def test_backward_multi_task():
    tasks = ["listen_plus", "like", "dislike", "listen_pct"]
    weights = {"listen_plus": 1.0, "like": 0.5, "dislike": 0.5, "listen_pct": 0.1}
    model = _build_dlrm(tasks=tasks)

    batch = _make_dummy_batch()
    out = model(batch)

    loss_fn = MultiTaskLoss(weights=weights)
    labels = {t: batch[t] for t in tasks}
    total_loss, task_losses = loss_fn(out, labels)

    total_loss.backward()
    assert len(task_losses) == 4
    assert not torch.isnan(total_loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_forward_gpu():
    model = _build_dlrm(device="cuda:0")

    batch = _make_dummy_batch(device="cuda:0")
    out = model(batch)

    assert out["listen_plus"].device.type == "cuda"
    assert out["listen_plus"].shape == (4,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_backward_gpu_bf16():
    tasks = ["listen_plus", "like"]
    model = _build_dlrm(device="cuda:0", tasks=tasks)

    batch = _make_dummy_batch(device="cuda:0")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
        loss_fn = MultiTaskLoss(weights={"listen_plus": 1.0, "like": 0.5})
        labels = {t: batch[t] for t in tasks}
        total_loss, _ = loss_fn(out, labels)

    total_loss.backward()
    assert not torch.isnan(total_loss)


def test_interaction_types():
    """Test all three interaction types produce valid output."""
    for itype in ["concat_mlp", "dcnv2"]:
        mc = ModelConfig(
            top_mlp_dims=[32, 16], bottom_mlp_dims=[32],
            interaction_type=itype,
        )
        model = _build_dlrm(model_config=mc)
        batch = _make_dummy_batch()
        out = model(batch)
        assert out["listen_plus"].shape == (4,)
        total = out["listen_plus"].sum()
        total.backward()


def test_forward_with_counters():
    model = _build_dlrm(num_counter_windows=1)
    batch = _make_dummy_batch(num_counter_windows=1)
    out = model(batch)
    assert "listen_plus" in out
    assert out["listen_plus"].shape == (4,)


def test_backward_with_counters():
    model = _build_dlrm(num_counter_windows=1)
    batch = _make_dummy_batch(num_counter_windows=1)
    out = model(batch)

    loss_fn = MultiTaskLoss(weights={"listen_plus": 1.0})
    labels = {"listen_plus": batch["listen_plus"]}
    total_loss, _ = loss_fn(out, labels)
    total_loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0
    assert not torch.isnan(total_loss)


def test_forward_cross_with_counters():
    model = _build_dlrm(num_counter_windows=2)
    batch = _make_dummy_batch(num_counter_windows=2)
    preds, cross_scores = model.forward_with_cross_scores(batch)

    assert "listen_plus" in preds
    assert preds["listen_plus"].shape == (4,)
    assert cross_scores.shape == (4, 4)


def test_forward_multi_window_counters():
    model = _build_dlrm(num_counter_windows=3)
    batch = _make_dummy_batch(num_counter_windows=3)
    out = model(batch)
    assert out["listen_plus"].shape == (4,)
    out["listen_plus"].sum().backward()


# ---------------------------------------------------------------------------
# OneTrans tests
# ---------------------------------------------------------------------------

def _onetrans_config(**overrides):
    ot_kw = {k: overrides.pop(k) for k in list(overrides) if k in TransformerConfig.__dataclass_fields__}
    return ModelConfig(
        model_type="onetrans", embedding_dim=16,
        transformer=TransformerConfig(d_model=32, n_heads=2, n_layers=2, ffn_mult=2,
                                      n_ns_tokens=4, **ot_kw),
        **overrides,
    )


def _build_onetrans(model_config=None, device="cpu", tasks=None, num_counter_windows=0):
    model_config = model_config or _onetrans_config()
    model_config.embedding_tables = list(_YAMBDA_EMBEDDING_TABLES)
    counter_days = list(range(1, num_counter_windows + 1)) if num_counter_windows > 0 else []
    full_config = Config()
    full_config.model = model_config
    full_config.feature = _yambda_feature_config(
        audio_dim=32,
        enable_counters=num_counter_windows > 0,
        counter_windows=counter_days if num_counter_windows > 0 else None,
    )
    full_config.data.schema = _YAMBDA_SCHEMA
    if tasks:
        full_config.train.loss_weights = {t: 1.0 for t in tasks}
    return OneTransModel(full_config, device=torch.device(device))


def test_onetrans_forward_cpu():
    model = _build_onetrans(model_config=_onetrans_config(attention_impl="sdpa"), tasks=["listen_plus"])
    batch = _make_dummy_batch()
    out = model(batch)
    assert "listen_plus" in out
    assert out["listen_plus"].shape == (4,)


def test_onetrans_backward_cpu():
    model = _build_onetrans(model_config=_onetrans_config(attention_impl="sdpa"), tasks=["listen_plus"])
    batch = _make_dummy_batch()
    out = model(batch)
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0
    assert not torch.isnan(loss)


def test_onetrans_no_pyramid():
    config = _onetrans_config(use_pyramid=False, attention_impl="sdpa")
    model = _build_onetrans(config)
    batch = _make_dummy_batch(batch_size=2, hist_len=5)
    out = model(batch)
    assert all(v.shape == (2,) for v in out.values())


def test_pyramid_schedule():
    schedule = pyramid_schedule(300, 8, 4)
    assert schedule[0] == 300
    assert schedule[-1] == 8
    assert all(schedule[i] >= schedule[i + 1] for i in range(len(schedule) - 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_onetrans_gpu_bf16():
    model = _build_onetrans(device="cuda:0", tasks=["listen_plus"])
    batch = _make_dummy_batch(device="cuda:0")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
        loss = sum(v.mean() for v in out.values())
    loss.backward()
    assert not torch.isnan(loss)


def test_onetrans_forward_with_counters():
    model = _build_onetrans(model_config=_onetrans_config(attention_impl="sdpa"), tasks=["listen_plus"], num_counter_windows=1)
    batch = _make_dummy_batch(num_counter_windows=1)
    out = model(batch)
    assert "listen_plus" in out
    assert out["listen_plus"].shape == (4,)


def test_onetrans_backward_with_counters():
    model = _build_onetrans(model_config=_onetrans_config(attention_impl="sdpa"), tasks=["listen_plus"], num_counter_windows=1)
    batch = _make_dummy_batch(num_counter_windows=1)
    out = model(batch)
    loss = out["listen_plus"].sum()
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0
    assert not torch.isnan(loss)


def test_onetrans_forward_cross():
    model = _build_onetrans(model_config=_onetrans_config(attention_impl="sdpa"), tasks=["listen_plus"])
    batch = _make_dummy_batch()
    preds, cross_scores = model.forward_with_cross_scores(batch)
    assert "listen_plus" in preds
    assert preds["listen_plus"].shape == (4,)
    assert cross_scores.shape == (4, 4)


def test_onetrans_forward_cross_with_counters():
    model = _build_onetrans(model_config=_onetrans_config(attention_impl="sdpa"), tasks=["listen_plus"], num_counter_windows=2)
    batch = _make_dummy_batch(num_counter_windows=2)
    preds, cross_scores = model.forward_with_cross_scores(batch)
    assert "listen_plus" in preds
    assert preds["listen_plus"].shape == (4,)
    assert cross_scores.shape == (4, 4)
    loss = preds["listen_plus"].sum() + cross_scores.sum()
    loss.backward()
    assert not torch.isnan(loss)


def test_dot_interaction_with_counters():
    """DotInteraction should handle variable feature dims from cross_proj."""
    mc = ModelConfig(
        top_mlp_dims=[32, 16], bottom_mlp_dims=[32],
        interaction_type="dot",
    )
    model = _build_dlrm(model_config=mc, num_counter_windows=1)
    batch = _make_dummy_batch(num_counter_windows=1)
    out = model(batch)
    assert out["listen_plus"].shape == (4,)
    out["listen_plus"].sum().backward()
