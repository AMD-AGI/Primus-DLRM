"""Tests for cross-feature hashing + Config.expand_cross_features() wiring."""
from __future__ import annotations

import pytest

from primus_dlrm.config import (
    Config,
    CrossFeatureSpec,
    DataConfig,
    EmbeddingTableConfig,
    FeatureConfig,
    ModelConfig,
    SchemaConfig,
)


# ---------------------------------------------------------------------------
# Hashing: bit-exact reproducibility, modulo behaviour, asymmetry, salt
# ---------------------------------------------------------------------------

def test_cross_hash_golden_vectors():
    """Golden values pinned to the closed-division spec.

    Computed once with xxhash 3.x via:
      xxhash.xxh64(seed=salt).update(struct.pack('<qq', a, b)).intdigest() % t

    Failure here means the wire format changed and breaks reproducibility
    across submitters / cache regeneration.
    """
    from primus_dlrm.data.hashing import cross_hash_int64

    assert cross_hash_int64(1, 2, 1000) == 524
    assert cross_hash_int64(1, 2, 1 << 30) == 1056167084
    assert cross_hash_int64(42, 7, 1_000_000) == 924376
    assert cross_hash_int64(0, 0, 1000) == 434
    assert cross_hash_int64(-1, -1, 1000) == 28


def test_cross_hash_modulo_in_range():
    from primus_dlrm.data.hashing import cross_hash_int64

    for table_size in (1, 7, 1024, 1_000_000):
        for a, b in [(0, 0), (1, 2), (123, 456), (1 << 40, 1 << 30)]:
            h = cross_hash_int64(a, b, table_size)
            assert 0 <= h < table_size, f"hash {h} out of range for size {table_size}"


def test_cross_hash_asymmetric():
    """Concat order matters: h(a, b) != h(b, a) for non-trivial (a, b)."""
    from primus_dlrm.data.hashing import cross_hash_int64

    assert cross_hash_int64(1, 2, 1000) != cross_hash_int64(2, 1, 1000)
    assert cross_hash_int64(42, 7, 1 << 30) != cross_hash_int64(7, 42, 1 << 30)


def test_cross_hash_salt_changes_output():
    from primus_dlrm.data.hashing import cross_hash_int64

    assert cross_hash_int64(1, 2, 1000, salt=0) != cross_hash_int64(1, 2, 1000, salt=5)


def test_cross_hash_batch_matches_scalar():
    import numpy as np

    from primus_dlrm.data.hashing import cross_hash_batch, cross_hash_int64

    arr_a = np.array([1, 42, 100, 0, -1], dtype=np.int64)
    arr_b = np.array([2, 7, 200, 0, -1], dtype=np.int64)
    table_size = 12345
    expected = np.array(
        [cross_hash_int64(int(a), int(b), table_size) for a, b in zip(arr_a, arr_b)],
        dtype=np.int64,
    )
    got = cross_hash_batch(arr_a, arr_b, table_size)
    np.testing.assert_array_equal(got, expected)


def test_cross_hash_batch_broadcasts_scalar_b():
    import numpy as np

    from primus_dlrm.data.hashing import cross_hash_batch, cross_hash_int64

    arr_a = np.array([1, 2, 3, 4], dtype=np.int64)
    arr_b = np.array([7], dtype=np.int64)
    table_size = 1000
    expected = np.array(
        [cross_hash_int64(int(a), 7, table_size) for a in arr_a], dtype=np.int64,
    )
    got = cross_hash_batch(arr_a, arr_b, table_size)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# n-way hashing: extends the 2-way wire format to >=2 keys
# ---------------------------------------------------------------------------

def test_cross_hash_nway_golden_vectors():
    """Pin n-way values for n=3 and n=4. Failure here means the n-way wire
    format changed and breaks reproducibility for any cross spec with >2 keys.
    """
    from primus_dlrm.data.hashing import cross_hash_nway

    assert cross_hash_nway([1, 2, 3], 1000) == 706
    assert cross_hash_nway([1, 2, 3], 1 << 30) == 633392634
    assert cross_hash_nway([42, 7, 15], 1_000_000) == 847584
    assert cross_hash_nway([1, 2, 3, 4], 1000) == 45


def test_cross_hash_nway_2way_matches_int64():
    """cross_hash_nway([a, b], t, s) must equal cross_hash_int64(a, b, t, s).

    Backward-compat invariant: the 2-way wire format is unchanged after
    introducing n-way support, so existing trained 2-way cross embeddings
    keep producing the same bucket ids.
    """
    from primus_dlrm.data.hashing import cross_hash_int64, cross_hash_nway

    cases = [(0, 0, 1000, 0), (1, 2, 1000, 0), (42, 7, 1_000_000, 0),
             (-1, -1, 1000, 0), (1, 2, 1000, 5), (1 << 40, 1 << 30, 1 << 30, 0)]
    for a, b, t, s in cases:
        assert cross_hash_nway([a, b], t, s) == cross_hash_int64(a, b, t, s)


def test_cross_hash_nway_order_matters():
    from primus_dlrm.data.hashing import cross_hash_nway

    assert cross_hash_nway([1, 2, 3], 1000) != cross_hash_nway([3, 2, 1], 1000)
    assert cross_hash_nway([3, 2, 1], 1000) == 313


def test_cross_hash_nway_salt_changes_output():
    from primus_dlrm.data.hashing import cross_hash_nway

    assert cross_hash_nway([1, 2, 3], 1000, salt=0) != cross_hash_nway([1, 2, 3], 1000, salt=5)


def test_cross_hash_nway_rejects_single_key():
    from primus_dlrm.data.hashing import cross_hash_nway

    with pytest.raises(AssertionError):
        cross_hash_nway([1], 1000)


# ---------------------------------------------------------------------------
# Config.expand_cross_features() — single source of truth, on/off flag
# ---------------------------------------------------------------------------

def _baseline_config() -> Config:
    """Minimal Config that mirrors the Yambda 5b shape but at toy sizes."""
    cfg = Config()
    cfg.model = ModelConfig(
        model_type="onetrans",
        embedding_dim=16,
        embedding_tables=[
            EmbeddingTableConfig("item", ["item"], num_embeddings=100),
            EmbeddingTableConfig("artist", ["artist"], num_embeddings=50),
            EmbeddingTableConfig("album", ["album"], num_embeddings=30),
            EmbeddingTableConfig("uid", ["uid"], num_embeddings=200),
        ],
    )
    cfg.feature = FeatureConfig(
        scalar_features=["uid", "item", "artist", "album"],
    )
    cfg.data = DataConfig()
    cfg.data.schema = SchemaConfig(
        batch_to_feature={
            "item_id": "item", "artist_id": "artist", "album_id": "album",
        },
        kjt_feature_order=["item", "artist", "album", "uid"],
    )
    return cfg


def test_expand_registers_enabled_specs_into_all_three_lists():
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="user_x_artist", keys=["uid", "artist_id"], num_embeddings=1024),
        CrossFeatureSpec(name="user_x_hour", keys=["uid", "hour_of_day"], num_embeddings=512),
    ]

    expanded = cfg.expand_cross_features()

    assert {s.name for s in expanded} == {"user_x_artist", "user_x_hour"}

    table_names = {t.name for t in cfg.model.embedding_tables}
    assert "user_x_artist" in table_names and "user_x_hour" in table_names

    # Cross tables inherit embedding_dim from model.embedding_dim via the
    # standard ``resolved_embedding_tables()`` pass.
    resolved = {t.name: t for t in cfg.model.resolved_embedding_tables()}
    assert resolved["user_x_artist"].embedding_dim == 16
    assert resolved["user_x_artist"].num_embeddings == 1024
    assert resolved["user_x_hour"].num_embeddings == 512

    scalar_names = cfg.feature.scalar_feature_names
    assert "user_x_artist" in scalar_names and "user_x_hour" in scalar_names

    assert cfg.data.schema.batch_to_feature["user_x_artist_id"] == "user_x_artist"
    assert cfg.data.schema.batch_to_feature["user_x_hour_id"] == "user_x_hour"
    assert "user_x_artist" in cfg.data.schema.kjt_feature_order
    assert "user_x_hour" in cfg.data.schema.kjt_feature_order


def test_expand_skips_disabled_specs():
    """enabled=False removes a cross from EVERY downstream list."""
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="user_x_artist", keys=["uid", "artist_id"], num_embeddings=1024),
        CrossFeatureSpec(name="user_x_album", keys=["uid", "album_id"], num_embeddings=512, enabled=False),
        CrossFeatureSpec(name="user_x_hour", keys=["uid", "hour_of_day"], num_embeddings=256),
    ]

    expanded = cfg.expand_cross_features()
    assert {s.name for s in expanded} == {"user_x_artist", "user_x_hour"}

    table_names = {t.name for t in cfg.model.embedding_tables}
    assert "user_x_album" not in table_names
    assert "user_x_album" not in cfg.feature.scalar_feature_names
    assert "user_x_album_id" not in cfg.data.schema.batch_to_feature
    assert "user_x_album" not in cfg.data.schema.kjt_feature_order


def test_expand_is_idempotent():
    """Calling expand_cross_features() twice is a no-op on the second call."""
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="user_x_artist", keys=["uid", "artist_id"], num_embeddings=1024),
    ]
    n_tables_before = len(cfg.model.embedding_tables)

    first = cfg.expand_cross_features()
    n_after_first = len(cfg.model.embedding_tables)
    second = cfg.expand_cross_features()
    n_after_second = len(cfg.model.embedding_tables)

    assert len(first) == 1
    assert len(second) == 0  # already registered
    assert n_after_first == n_tables_before + 1
    assert n_after_second == n_after_first  # no double registration


def test_expand_rejects_collision_with_native_table():
    cfg = _baseline_config()
    cfg.model.cross_features = [
        # 'item' is already a native table
        CrossFeatureSpec(name="item", keys=["uid", "item_id"], num_embeddings=1024),
    ]
    with pytest.raises(ValueError, match="collides"):
        cfg.expand_cross_features()


def test_expand_rejects_unknown_key():
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="bad", keys=["uid", "not_a_real_key"], num_embeddings=1024),
    ]
    with pytest.raises(ValueError, match="unknown key"):
        cfg.expand_cross_features()


def test_expand_rejects_zero_buckets():
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="user_x_artist", keys=["uid", "artist_id"], num_embeddings=0),
    ]
    with pytest.raises(ValueError, match="num_embeddings"):
        cfg.expand_cross_features()


def test_expand_rejects_single_key():
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="bad", keys=["uid"], num_embeddings=1024),
    ]
    with pytest.raises(ValueError, match="at least 2 keys"):
        cfg.expand_cross_features()


def test_expand_accepts_3way_spec():
    """A 3-key cross spec must register cleanly into all three downstream lists."""
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(
            name="user_x_artist_x_hour",
            keys=["uid", "artist_id", "hour_of_day"],
            num_embeddings=4096,
        ),
    ]
    expanded = cfg.expand_cross_features()

    assert {s.name for s in expanded} == {"user_x_artist_x_hour"}

    table_names = {t.name for t in cfg.model.embedding_tables}
    assert "user_x_artist_x_hour" in table_names
    assert "user_x_artist_x_hour" in cfg.feature.scalar_feature_names
    assert cfg.data.schema.batch_to_feature["user_x_artist_x_hour_id"] == "user_x_artist_x_hour"
    assert "user_x_artist_x_hour" in cfg.data.schema.kjt_feature_order


def test_expand_appends_in_spec_order():
    """NS-token concat order = cross-spec order; this is the contract."""
    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="z_first", keys=["uid", "artist_id"], num_embeddings=8),
        CrossFeatureSpec(name="a_second", keys=["uid", "album_id"], num_embeddings=8),
        CrossFeatureSpec(name="m_third", keys=["uid", "hour_of_day"], num_embeddings=8),
    ]
    cfg.expand_cross_features()

    # Native scalars come first (uid, item, artist, album), then crosses in
    # spec order. The model relies on this for deterministic NS concat.
    scalars = cfg.feature.scalar_feature_names
    cross_in_scalars = [s for s in scalars if s.startswith(("z_", "a_", "m_"))]
    assert cross_in_scalars == ["z_first", "a_second", "m_third"]


def test_load_mmap_missing_flat_is_organic_raises(tmp_path):
    """Cache contract: flat_is_organic.npy is required by FlatEventStore.load_mmap.

    Caches built before the cross-feature change won't have this file; running
    the new code against such a cache MUST fail loudly at load time (not
    silently or at first __getitem__).
    """
    import json
    import numpy as np

    from primus_dlrm.data.dataset import FlatEventStore

    n = 8
    cache_dir = tmp_path / "fake_cache"
    cache_dir.mkdir()
    # Write all required files EXCEPT flat_is_organic.npy.
    for name in ("flat_uid", "flat_item_ids", "flat_timestamps",
                 "flat_event_types", "flat_played_ratio",
                 "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
                 "user_start", "user_end", "unique_uids"):
        np.save(cache_dir / f"{name}.npy", np.zeros(n, dtype=np.int64))
    with open(cache_dir / "store_meta.json", "w") as f:
        json.dump({
            "num_users": 1, "total_events": n,
            "enable_counters": False, "counter_windows_days": [],
        }, f)

    with pytest.raises(FileNotFoundError, match="flat_is_organic"):
        FlatEventStore.load_mmap(cache_dir)


def test_load_yaml_strategy_a_config():
    """End-to-end: loading the strategy_a YAML auto-registers all 7 crosses on load."""
    from pathlib import Path

    cfg_path = (
        Path(__file__).resolve().parent.parent
        / "configs" / "bench_onetrans_large_5b_strategy_a.yaml"
    )
    if not cfg_path.exists():
        pytest.skip(f"config not present: {cfg_path}")

    cfg = Config.load(cfg_path)

    expected_crosses = {
        "user_x_artist", "user_x_album", "user_x_hour",
        "item_x_hour", "artist_x_hour", "user_x_is_organic",
        "user_x_artist_x_hour",
    }
    table_names = {t.name for t in cfg.model.embedding_tables}
    assert expected_crosses.issubset(table_names)
    assert expected_crosses.issubset(set(cfg.feature.scalar_feature_names))
    for name in expected_crosses:
        assert cfg.data.schema.batch_to_feature[f"{name}_id"] == name
        assert name in cfg.data.schema.kjt_feature_order


# ---------------------------------------------------------------------------
# Dataset wiring: ScoringPair carries cross_ids, collate produces flat keys
# ---------------------------------------------------------------------------

def test_collate_to_dict_emits_cross_ids_per_spec():
    """collate_to_dict produces one tensor per cross spec name."""
    import torch

    from primus_dlrm.data.dataset import ScoringPair, collate_to_dict

    L = 4
    audio_d = 32

    def mk(uid, item, artist, album, cross):
        return ScoringPair(
            hist_lp_item_ids=torch.zeros(L, dtype=torch.long),
            hist_lp_artist_ids=torch.zeros(L, dtype=torch.long),
            hist_lp_album_ids=torch.zeros(L, dtype=torch.long),
            hist_like_item_ids=torch.zeros(L, dtype=torch.long),
            hist_like_artist_ids=torch.zeros(L, dtype=torch.long),
            hist_like_album_ids=torch.zeros(L, dtype=torch.long),
            hist_skip_item_ids=torch.zeros(L, dtype=torch.long),
            hist_skip_artist_ids=torch.zeros(L, dtype=torch.long),
            hist_skip_album_ids=torch.zeros(L, dtype=torch.long),
            uid=uid, item_id=item, artist_id=artist, album_id=album,
            audio_embed=torch.zeros(audio_d, dtype=torch.float32),
            listen_plus=0.0, like=0.0, dislike=0.0, listen_pct=0.0,
            cross_ids=cross,
        )

    batch = [
        mk(1, 10, 100, 1000, {"user_x_artist": 5, "user_x_hour": 2}),
        mk(2, 20, 200, 2000, {"user_x_artist": 7, "user_x_hour": 9}),
    ]
    out = collate_to_dict(batch)

    assert "user_x_artist_id" in out
    assert "user_x_hour_id" in out
    assert out["user_x_artist_id"].tolist() == [5, 7]
    assert out["user_x_hour_id"].tolist() == [2, 9]


def test_pipeline_batch_explodes_cross_ids():
    """collate_pipeline_batch routes cross_ids into a real KJT key per spec."""
    import torch

    pytest.importorskip("torchrec")
    from primus_dlrm.data.dataset import ScoringPair
    from primus_dlrm.data.pipeline_batch import collate_pipeline_batch

    cfg = _baseline_config()
    cfg.model.cross_features = [
        CrossFeatureSpec(name="user_x_artist", keys=["uid", "artist_id"], num_embeddings=1024),
    ]
    cfg.expand_cross_features()

    L = 4
    audio_d = 32

    def mk(uid, cross):
        return ScoringPair(
            hist_lp_item_ids=torch.zeros(L, dtype=torch.long),
            hist_lp_artist_ids=torch.zeros(L, dtype=torch.long),
            hist_lp_album_ids=torch.zeros(L, dtype=torch.long),
            hist_like_item_ids=torch.zeros(L, dtype=torch.long),
            hist_like_artist_ids=torch.zeros(L, dtype=torch.long),
            hist_like_album_ids=torch.zeros(L, dtype=torch.long),
            hist_skip_item_ids=torch.zeros(L, dtype=torch.long),
            hist_skip_artist_ids=torch.zeros(L, dtype=torch.long),
            hist_skip_album_ids=torch.zeros(L, dtype=torch.long),
            uid=uid, item_id=10, artist_id=100, album_id=1000,
            audio_embed=torch.zeros(audio_d, dtype=torch.float32),
            listen_plus=0.0, like=0.0, dislike=0.0, listen_pct=0.0,
            cross_ids=cross,
        )

    batch = [mk(1, {"user_x_artist": 5}), mk(2, {"user_x_artist": 7})]
    pb = collate_pipeline_batch(batch, cfg)

    # The cross-feature key lands in tensors keyed by "<spec>_id" and the KJT
    # carries it under the EC feature name.
    assert "user_x_artist_id" in pb.tensors
    assert pb.tensors["user_x_artist_id"].tolist() == [5, 7]
    assert "user_x_artist" in pb.unpooled_kjt.keys()
