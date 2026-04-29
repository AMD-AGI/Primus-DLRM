"""PyTorch Datasets for Yambda: flat (user, item) scoring pairs with split history."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from primus_dlrm.config import DataConfig
from primus_dlrm.data.preprocessing import EVENT_TYPE_MAP


# ---------------------------------------------------------------------------
# Path layout convention
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataPaths:
    """All on-disk locations for one dataset variant, derived from the data
    root and dataset size.

    The convention is::

        <data_root>/raw/<size>/multi_event.parquet           (size-specific raw events)
        <data_root>/shared_metadata/embeddings.parquet       (size-invariant catalog)
        <data_root>/shared_metadata/{artist,album}_item_mapping.parquet
        <data_root>/processed[_<size>]/                      (size-specific processed outputs)
        <data_root>/cache[_<size>]/                          (size-specific FlatEventStore cache)

    The 50m variant uses the legacy unsuffixed ``processed/`` and ``cache/``
    paths to stay backward compatible with existing on-disk data; 500m and 5b
    use ``processed_500m/`` / ``cache_500m/`` etc.
    """
    data_root: Path
    dataset_size: str

    @classmethod
    def from_config(cls, config: DataConfig, data_root: Path | str | None = None) -> "DataPaths":
        """Build paths from a DataConfig. ``data_root`` (CLI) overrides
        ``config.data_dir`` when provided."""
        root = Path(data_root) if data_root is not None else Path(config.data_dir)
        return cls(data_root=root, dataset_size=config.dataset_size)

    @property
    def raw(self) -> Path:
        return self.data_root / "raw" / self.dataset_size

    @property
    def metadata(self) -> Path:
        return self.data_root / "shared_metadata"

    @property
    def processed(self) -> Path:
        if self.dataset_size == "50m":
            return self.data_root / "processed"
        return self.data_root / f"processed_{self.dataset_size}"

    @property
    def cache(self) -> Path:
        if self.dataset_size == "50m":
            return self.data_root / "cache"
        return self.data_root / f"cache_{self.dataset_size}"

logger = logging.getLogger(__name__)

LISTEN_TYPE = EVENT_TYPE_MAP["listen"]
LIKE_TYPE = EVENT_TYPE_MAP["like"]
DISLIKE_TYPE = EVENT_TYPE_MAP["dislike"]
LISTEN_PLUS_THRESHOLD = 50
SECONDS_PER_DAY = 86400


@dataclass
class ScoringPair:
    """One (user, item) scoring pair with split-history features."""
    hist_lp_item_ids: torch.Tensor       # [L] listen+ item IDs
    hist_lp_artist_ids: torch.Tensor     # [L]
    hist_lp_album_ids: torch.Tensor      # [L]
    hist_like_item_ids: torch.Tensor     # [L] liked item IDs
    hist_like_artist_ids: torch.Tensor   # [L]
    hist_like_album_ids: torch.Tensor    # [L]
    hist_skip_item_ids: torch.Tensor     # [L] skipped item IDs
    hist_skip_artist_ids: torch.Tensor   # [L]
    hist_skip_album_ids: torch.Tensor    # [L]
    uid: int

    item_id: int
    artist_id: int
    album_id: int
    audio_embed: torch.Tensor            # [D_audio]

    listen_plus: float
    like: float
    dislike: float
    listen_pct: float

    user_counters: torch.Tensor | None = None    # [3W]
    item_counters: torch.Tensor | None = None    # [3W]
    cross_counters: torch.Tensor | None = None   # [9W]


class FlatEventStore:
    """Memory-efficient flat array storage for all user events.

    Supports optional memory-mapped backing files for 5B-scale datasets
    via ``save_mmap`` / ``load_mmap``.
    """

    def __init__(
        self,
        sessions_df: pl.DataFrame,
        enable_counters: bool = False,
        item_to_artist: np.ndarray | None = None,
        item_to_album: np.ndarray | None = None,
        counter_windows_days: list[int] | None = None,
    ):
        logger.info("Building flat event store from sessions...")

        sorted_sessions = sessions_df.sort(["uid", "session_id"])
        unique_uids = np.unique(sorted_sessions["uid"].to_numpy())
        self.num_users = len(unique_uids)

        exploded = sorted_sessions.explode([
            "item_ids", "timestamps", "event_types",
            "is_organic", "played_ratio_pct", "track_length_seconds",
        ])

        self.flat_uid = exploded["uid"].to_numpy().astype(np.int64)
        self.flat_item_ids = exploded["item_ids"].to_numpy().astype(np.int64)
        self.flat_timestamps = exploded["timestamps"].to_numpy().astype(np.int64)
        self.flat_event_types = exploded["event_types"].to_numpy().astype(np.int64)
        self.flat_played_ratio = exploded["played_ratio_pct"].to_numpy().astype(np.float32)

        np.nan_to_num(self.flat_played_ratio, copy=False, nan=0.0)

        is_listen = self.flat_event_types == LISTEN_TYPE
        self.flat_is_listen_plus = is_listen & (self.flat_played_ratio >= LISTEN_PLUS_THRESHOLD)
        self.flat_is_like = self.flat_event_types == LIKE_TYPE
        self.flat_is_skip = is_listen & (self.flat_played_ratio < LISTEN_PLUS_THRESHOLD)

        uid_changes = np.where(np.diff(self.flat_uid) != 0)[0] + 1
        starts = np.concatenate([[0], uid_changes])
        ends = np.concatenate([uid_changes, [len(self.flat_uid)]])
        uid_vals = self.flat_uid[starts]

        max_uid = int(uid_vals.max()) + 1
        self.user_start = np.full(max_uid, -1, dtype=np.int64)
        self.user_end = np.full(max_uid, -1, dtype=np.int64)
        self.user_start[uid_vals] = starts
        self.user_end[uid_vals] = ends

        self.unique_uids = uid_vals
        self.total_events = len(self.flat_item_ids)
        logger.info(f"Flat event store: {self.total_events:,} events, {self.num_users:,} users")

        self.enable_counters = enable_counters
        self.counter_windows_days = counter_windows_days or []
        if enable_counters:
            assert item_to_artist is not None and item_to_album is not None
            self._precompute_counters(item_to_artist, item_to_album)

    def _precompute_counters(
        self, item_to_artist: np.ndarray, item_to_album: np.ndarray,
    ) -> None:
        """Fully precompute all windowed counters using the numba-parallel
        sliding-window kernels in ``primus_dlrm.data.counters``."""
        from primus_dlrm.data import counters  # noqa: WPS433  (lazy import for numba)

        W = len(self.counter_windows_days)
        N = self.total_events
        logger.info(
            f"Precomputing counters: {N:,} events, {W} windows "
            f"{self.counter_windows_days}"
        )

        self.user_counters = counters.precompute_user_counters(
            self.flat_uid, self.flat_timestamps,
            self.flat_is_listen_plus, self.flat_is_like, self.flat_is_skip,
            self.counter_windows_days,
        )
        self.item_counters = counters.precompute_item_counters(
            self.flat_item_ids, self.flat_timestamps,
            self.flat_is_listen_plus, self.flat_is_like, self.flat_is_skip,
            self.counter_windows_days,
        )
        self.cross_counters = counters.precompute_cross_counters(
            self.flat_uid, self.flat_item_ids, self.flat_timestamps,
            self.flat_is_listen_plus, self.flat_is_like, self.flat_is_skip,
            item_to_artist, item_to_album,
            self.counter_windows_days,
        )

        logger.info(
            f"Counter precomputation done. Memory: "
            f"user={self.user_counters.nbytes / 1e9:.2f}GB, "
            f"item={self.item_counters.nbytes / 1e9:.2f}GB, "
            f"cross={self.cross_counters.nbytes / 1e9:.2f}GB"
        )

    def get_user_slice(self, uid: int) -> tuple[int, int]:
        return int(self.user_start[uid]), int(self.user_end[uid])

    # ------------------------------------------------------------------
    # mmap persistence for large datasets
    # ------------------------------------------------------------------

    def save_mmap(self, mmap_dir: str | Path) -> None:
        """Persist flat arrays to disk as memory-mapped files."""
        mmap_dir = Path(mmap_dir)
        mmap_dir.mkdir(parents=True, exist_ok=True)

        for name in ("flat_uid", "flat_item_ids", "flat_timestamps",
                      "flat_event_types", "flat_played_ratio",
                      "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
                      "user_start", "user_end", "unique_uids"):
            arr = getattr(self, name)
            np.save(mmap_dir / f"{name}.npy", arr)

        if self.enable_counters:
            for name in ("user_counters", "item_counters", "cross_counters"):
                np.save(mmap_dir / f"{name}.npy", getattr(self, name))

        meta = {
            "num_users": self.num_users,
            "total_events": self.total_events,
            "enable_counters": self.enable_counters,
            "counter_windows_days": self.counter_windows_days,
        }
        import json
        with open(mmap_dir / "store_meta.json", "w") as f:
            json.dump(meta, f)
        logger.info(f"Saved mmap store to {mmap_dir}")

    @classmethod
    def load_mmap(cls, mmap_dir: str | Path, in_memory: bool = False) -> "FlatEventStore":
        """Load flat arrays from cache files.

        Args:
            mmap_dir: directory containing the .npy cache files.
            in_memory: if True, load arrays fully into RAM (fast random
                access, higher memory). If False, use memory-mapped files
                (lazy page faults, lower memory but slower on networked FS).
        """
        mmap_dir = Path(mmap_dir)
        import json
        with open(mmap_dir / "store_meta.json") as f:
            meta = json.load(f)

        store = object.__new__(cls)
        store.num_users = meta["num_users"]
        store.total_events = meta["total_events"]
        store.enable_counters = meta["enable_counters"]
        store.counter_windows_days = meta.get("counter_windows_days", [])

        for name in ("flat_uid", "flat_item_ids", "flat_timestamps",
                      "flat_event_types", "flat_played_ratio",
                      "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
                      "user_start", "user_end", "unique_uids"):
            setattr(store, name, np.load(
                mmap_dir / f"{name}.npy", mmap_mode="r",
            ))

        if store.enable_counters:
            for name in ("user_counters", "item_counters", "cross_counters"):
                setattr(store, name, np.load(
                    mmap_dir / f"{name}.npy", mmap_mode="r",
                ))

        logger.info(
            f"Loaded mmap store: {store.total_events:,} events, "
            f"{store.num_users:,} users"
        )
        return store


def _cache_key(split: str, counter_windows_days: list[int] | None) -> str:
    """Deterministic cache subdirectory name."""
    wins = "_".join(str(d) for d in (counter_windows_days or []))
    return f"{split}_w{wins}" if wins else split


def _try_load_cached_store(
    config: DataConfig, paths: "DataPaths", split: str,
    in_memory: bool = False,
) -> FlatEventStore | None:
    """Return cached FlatEventStore if available and use_cache is enabled."""
    if not config.use_cache:
        return None
    cache_dir = paths.cache / _cache_key(
        split, config.counter_windows_days if config.enable_counters else None,
    )
    meta_path = cache_dir / "store_meta.json"
    if not meta_path.exists():
        return None
    try:
        store = FlatEventStore.load_mmap(cache_dir, in_memory=in_memory)
        if store.enable_counters != config.enable_counters:
            logger.info("Cache counter mismatch, rebuilding...")
            return None
        return store
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}, rebuilding...")
        return None


def _save_store_cache(
    store: FlatEventStore, config: DataConfig, paths: "DataPaths", split: str,
) -> None:
    if not config.use_cache:
        return
    cache_dir = paths.cache / _cache_key(
        split, config.counter_windows_days if config.enable_counters else None,
    )
    store.save_mmap(cache_dir)


class YambdaTrainDataset(Dataset):
    """Training dataset with split history pools (listen+, like, skip)."""

    def __init__(self, config: DataConfig, paths: DataPaths):
        self.config = config
        self.paths = paths

        logger.info(
            f"Loading preprocessed data from {paths.processed} "
            f"(metadata: {paths.metadata}, cache: {paths.cache})"
        )
        self.item_popularity = np.load(paths.processed / "item_popularity.npy")

        with open(paths.processed / "split_meta.json") as f:
            self.split_meta = json.load(f)

        self._load_metadata(paths.metadata)

        cached = _try_load_cached_store(config, paths, "train", in_memory=False)
        if cached is not None:
            self.store = cached
        else:
            sessions = pl.read_parquet(paths.processed / "train_sessions.parquet")
            self.store = FlatEventStore(
                sessions,
                enable_counters=config.enable_counters,
                item_to_artist=self.item_to_artist if config.enable_counters else None,
                item_to_album=self.item_to_album if config.enable_counters else None,
                counter_windows_days=config.counter_windows_days if config.enable_counters else None,
            )
            _save_store_cache(self.store, config, paths, "train")

        self._build_sample_positions()

        logger.info(
            f"Dataset ready: {len(self._positions)} training positions, "
            f"{self.num_items} items, {self.store.num_users} users"
        )

    def _load_metadata(self, metadata_dir: Path) -> None:
        artist_map = pl.read_parquet(metadata_dir / "artist_item_mapping.parquet")
        self.item_to_artist = np.zeros(self.item_popularity.shape[0], dtype=np.int64)
        valid = artist_map.filter(pl.col("item_id") < len(self.item_to_artist))
        self.item_to_artist[valid["item_id"].to_numpy()] = valid["artist_id"].to_numpy()
        self.num_artists = int(artist_map["artist_id"].max()) + 1

        album_map = pl.read_parquet(metadata_dir / "album_item_mapping.parquet")
        self.item_to_album = np.zeros(self.item_popularity.shape[0], dtype=np.int64)
        valid = album_map.filter(pl.col("item_id") < len(self.item_to_album))
        self.item_to_album[valid["item_id"].to_numpy()] = valid["album_id"].to_numpy()
        self.num_albums = int(album_map["album_id"].max()) + 1

        emb_path = metadata_dir / "embeddings.parquet"
        if emb_path.exists():
            emb_df = pl.read_parquet(emb_path)
            item_ids = emb_df["item_id"].to_numpy()
            embeds = np.stack(emb_df["normalized_embed"].to_numpy())
            self.audio_dim = embeds.shape[1]
            self.audio_embeddings = np.zeros(
                (self.item_popularity.shape[0], self.audio_dim), dtype=np.float32
            )
            valid_mask = item_ids < len(self.audio_embeddings)
            self.audio_embeddings[item_ids[valid_mask]] = embeds[valid_mask]
        else:
            self.audio_dim = 256
            self.audio_embeddings = np.zeros(
                (self.item_popularity.shape[0], self.audio_dim), dtype=np.float32
            )

        self.num_items = len(self.item_popularity)

    @property
    def vocab_sizes(self) -> list[int]:
        """Vocab sizes in schema table order (item, artist, album, uid)."""
        return [
            self.num_items,
            self.num_artists,
            self.num_albums,
            int(self.store.unique_uids.max()) + 1,
        ]

    def _build_sample_positions(self) -> None:
        """Build or load flat array of valid training positions.

        If a cached ``positions_L{history_length}.npy`` exists in the cache
        dir, it is memory-mapped (shared across ranks, zero computation).
        Otherwise, positions are computed vectorised and optionally saved.
        """
        L = self.config.history_length
        cache_key = _cache_key(
            "train",
            self.config.counter_windows_days if self.config.enable_counters else None,
        )
        pos_path = self.paths.cache / cache_key / f"positions_L{L}.npy"

        if pos_path.exists():
            self._positions = np.load(pos_path)
            logger.info(
                f"Valid training positions: {len(self._positions):,} (from cache)"
            )
            return

        starts = self.store.user_start
        ends = self.store.user_end
        counts = np.maximum(ends - starts - L, 0)
        total = int(counts.sum())
        positions = np.empty(total, dtype=np.int64)
        offset = 0
        for i in range(len(starts)):
            n = int(counts[i])
            if n > 0:
                positions[offset:offset + n] = np.arange(
                    int(starts[i]) + L, int(ends[i]), dtype=np.int64,
                )
                offset += n

        if self.config.use_cache:
            pos_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pos_path, positions)
            logger.info(f"Saved positions cache: {pos_path}")
            self._positions = positions
        else:
            self._positions = positions
        logger.info(f"Valid training positions: {len(self._positions):,}")

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int) -> ScoringPair:
        flat_pos = self._positions[idx]
        uid = int(self.store.flat_uid[flat_pos])
        user_start, _ = self.store.get_user_slice(uid)

        history = self._get_split_history(flat_pos, user_start)

        target_item = int(self.store.flat_item_ids[flat_pos])
        event_type = int(self.store.flat_event_types[flat_pos])
        played_ratio = float(self.store.flat_played_ratio[flat_pos])

        audio = self.audio_embeddings[target_item] if target_item < len(self.audio_embeddings) else np.zeros(self.audio_dim, dtype=np.float32)

        counters_kwargs = {}
        if self.store.enable_counters:
            counters_kwargs["user_counters"] = torch.from_numpy(
                self.store.user_counters[flat_pos].copy()
            )
            counters_kwargs["item_counters"] = torch.from_numpy(
                self.store.item_counters[flat_pos].copy()
            )
            counters_kwargs["cross_counters"] = torch.from_numpy(
                self.store.cross_counters[flat_pos].copy()
            )

        return ScoringPair(
            hist_lp_item_ids=torch.from_numpy(history["lp_items"].copy()),
            hist_lp_artist_ids=torch.from_numpy(history["lp_artists"].copy()),
            hist_lp_album_ids=torch.from_numpy(history["lp_albums"].copy()),
            hist_like_item_ids=torch.from_numpy(history["like_items"].copy()),
            hist_like_artist_ids=torch.from_numpy(history["like_artists"].copy()),
            hist_like_album_ids=torch.from_numpy(history["like_albums"].copy()),
            hist_skip_item_ids=torch.from_numpy(history["skip_items"].copy()),
            hist_skip_artist_ids=torch.from_numpy(history["skip_artists"].copy()),
            hist_skip_album_ids=torch.from_numpy(history["skip_albums"].copy()),
            uid=uid,
            item_id=target_item,
            artist_id=int(self.item_to_artist[target_item]) if target_item < len(self.item_to_artist) else 0,
            album_id=int(self.item_to_album[target_item]) if target_item < len(self.item_to_album) else 0,
            audio_embed=torch.from_numpy(audio.copy()),
            listen_plus=1.0 if (event_type == LISTEN_TYPE and played_ratio >= LISTEN_PLUS_THRESHOLD) else 0.0,
            like=1.0 if event_type == LIKE_TYPE else 0.0,
            dislike=1.0 if event_type == DISLIKE_TYPE else 0.0,
            listen_pct=min(played_ratio / 100.0, 1.0) if event_type == LISTEN_TYPE else 0.0,
            **counters_kwargs,
        )

    def _get_split_history(self, flat_pos: int, user_start: int) -> dict:
        """Scan backward from flat_pos, split events into listen+/like/skip pools."""
        L = self.config.history_length
        scan_end = flat_pos
        scan_start = max(user_start, flat_pos - 500)

        item_ids = self.store.flat_item_ids[scan_start:scan_end]
        is_lp = self.store.flat_is_listen_plus[scan_start:scan_end]
        is_like = self.store.flat_is_like[scan_start:scan_end]
        is_skip = self.store.flat_is_skip[scan_start:scan_end]

        lp_items = item_ids[is_lp][-L:] if is_lp.any() else np.array([], dtype=np.int64)
        like_items = item_ids[is_like][-L:] if is_like.any() else np.array([], dtype=np.int64)
        skip_items = item_ids[is_skip][-L:] if is_skip.any() else np.array([], dtype=np.int64)

        def pad_and_lookup(items: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            n = len(items)
            padded = np.zeros(L, dtype=np.int64)
            if n > 0:
                padded[L - n:] = items
            artists = self.item_to_artist[padded]
            albums = self.item_to_album[padded]
            return padded, artists, albums

        lp_i, lp_a, lp_al = pad_and_lookup(lp_items)
        like_i, like_a, like_al = pad_and_lookup(like_items)
        skip_i, skip_a, skip_al = pad_and_lookup(skip_items)

        return {
            "lp_items": lp_i, "lp_artists": lp_a, "lp_albums": lp_al,
            "like_items": like_i, "like_artists": like_a, "like_albums": like_al,
            "skip_items": skip_i, "skip_artists": skip_a, "skip_albums": skip_al,
        }


class YambdaEvalDataset(Dataset):
    """Evaluation dataset with split history pools."""

    def __init__(self, config: DataConfig, paths: DataPaths):
        self.config = config
        self.paths = paths

        test_events = pl.read_parquet(paths.processed / "test_events.parquet")
        self.item_popularity = np.load(paths.processed / "item_popularity.npy")

        self._load_metadata(paths.metadata)

        cached = _try_load_cached_store(config, paths, "eval")
        if cached is not None:
            self.store = cached
        else:
            sessions = pl.read_parquet(paths.processed / "train_sessions.parquet")
            self.store = FlatEventStore(
                sessions,
                enable_counters=config.enable_counters,
                item_to_artist=self.item_to_artist if config.enable_counters else None,
                item_to_album=self.item_to_album if config.enable_counters else None,
                counter_windows_days=config.counter_windows_days if config.enable_counters else None,
            )
            _save_store_cache(self.store, config, paths, "eval")

        if config.enable_counters:
            self._precompute_eval_item_counters()

        self.test_ground_truth: dict[int, dict] = {}
        for row in test_events.iter_rows(named=True):
            uid = row["uid"]
            if uid not in self.test_ground_truth:
                self.test_ground_truth[uid] = {"listen_plus": set(), "like": set()}

            event_type = row["event_type"]
            item_id = row["item_id"]
            played = row.get("played_ratio_pct")

            if event_type == LISTEN_TYPE and played is not None and played >= LISTEN_PLUS_THRESHOLD:
                self.test_ground_truth[uid]["listen_plus"].add(item_id)
            if event_type == LIKE_TYPE:
                self.test_ground_truth[uid]["like"].add(item_id)

        self.test_uids = sorted(self.test_ground_truth.keys())
        logger.info(f"Eval dataset: {len(self.test_uids)} test users")

    def _load_metadata(self, metadata_dir: Path) -> None:
        artist_map = pl.read_parquet(metadata_dir / "artist_item_mapping.parquet")
        self.item_to_artist = np.zeros(len(self.item_popularity), dtype=np.int64)
        valid = artist_map.filter(pl.col("item_id") < len(self.item_to_artist))
        self.item_to_artist[valid["item_id"].to_numpy()] = valid["artist_id"].to_numpy()

        album_map = pl.read_parquet(metadata_dir / "album_item_mapping.parquet")
        self.item_to_album = np.zeros(len(self.item_popularity), dtype=np.int64)
        valid = album_map.filter(pl.col("item_id") < len(self.item_to_album))
        self.item_to_album[valid["item_id"].to_numpy()] = valid["album_id"].to_numpy()

        emb_path = metadata_dir / "embeddings.parquet"
        if emb_path.exists():
            emb_df = pl.read_parquet(emb_path)
            item_ids = emb_df["item_id"].to_numpy()
            embeds = np.stack(emb_df["normalized_embed"].to_numpy())
            self.audio_dim = embeds.shape[1]
            self.audio_embeddings = np.zeros(
                (len(self.item_popularity), self.audio_dim), dtype=np.float32
            )
            valid_mask = item_ids < len(self.audio_embeddings)
            self.audio_embeddings[item_ids[valid_mask]] = embeds[valid_mask]
        else:
            self.audio_dim = 256
            self.audio_embeddings = np.zeros(
                (len(self.item_popularity), self.audio_dim), dtype=np.float32
            )

    def get_user_history(self, uid: int) -> dict:
        """Get split history for a test user (at end of training period)."""
        L = self.config.history_length
        start, end = self.store.get_user_slice(uid)
        if start < 0:
            z = np.zeros(L, dtype=np.int64)
            return {k: z.copy() for k in [
                "lp_items", "lp_artists", "lp_albums",
                "like_items", "like_artists", "like_albums",
                "skip_items", "skip_artists", "skip_albums",
            ]}

        scan_start = max(start, end - 500)
        item_ids = self.store.flat_item_ids[scan_start:end]
        is_lp = self.store.flat_is_listen_plus[scan_start:end]
        is_like = self.store.flat_is_like[scan_start:end]
        is_skip = self.store.flat_is_skip[scan_start:end]

        def pad_and_lookup(mask):
            items = item_ids[mask][-L:] if mask.any() else np.array([], dtype=np.int64)
            n = len(items)
            padded = np.zeros(L, dtype=np.int64)
            if n > 0:
                padded[L - n:] = items
            artists = self.item_to_artist[padded]
            albums = self.item_to_album[padded]
            return padded, artists, albums

        lp_i, lp_a, lp_al = pad_and_lookup(is_lp)
        like_i, like_a, like_al = pad_and_lookup(is_like)
        skip_i, skip_a, skip_al = pad_and_lookup(is_skip)

        return {
            "lp_items": lp_i, "lp_artists": lp_a, "lp_albums": lp_al,
            "like_items": like_i, "like_artists": like_a, "like_albums": like_al,
            "skip_items": skip_i, "skip_artists": skip_a, "skip_albums": skip_al,
        }

    def get_user_train_items(self, uid: int) -> np.ndarray:
        """Return all unique item_ids this user interacted with during training."""
        start, end = self.store.get_user_slice(uid)
        if start < 0:
            return np.array([], dtype=np.int64)
        return np.unique(self.store.flat_item_ids[start:end])

    def _precompute_eval_item_counters(self) -> None:
        """Precompute item counters at end-of-training timestamp for all items."""
        W = len(self.config.counter_windows_days)
        num_items = len(self.item_popularity)
        self.eval_item_counters = np.zeros((num_items, 3 * W), dtype=np.float32)

        max_ts = int(self.store.flat_timestamps.max())
        sort_idx = np.lexsort((self.store.flat_timestamps, self.store.flat_item_ids))
        sorted_items = self.store.flat_item_ids[sort_idx]
        sorted_ts = self.store.flat_timestamps[sort_idx]
        sorted_is_lp = self.store.flat_is_listen_plus[sort_idx]
        sorted_is_like = self.store.flat_is_like[sort_idx]
        sorted_is_skip = self.store.flat_is_skip[sort_idx]

        item_changes = np.where(np.diff(sorted_items) != 0)[0] + 1
        item_starts = np.concatenate([[0], item_changes])
        item_ends = np.concatenate([item_changes, [len(sorted_items)]])

        for w_idx, w_days in enumerate(self.config.counter_windows_days):
            window_start_ts = max_ts - w_days * SECONDS_PER_DAY
            col_lp = w_idx * 3
            col_like = w_idx * 3 + 1
            col_skip = w_idx * 3 + 2

            for seg_idx in range(len(item_starts)):
                s, e = int(item_starts[seg_idx]), int(item_ends[seg_idx])
                iid = int(sorted_items[s])
                mask = sorted_ts[s:e] >= window_start_ts
                if not mask.any():
                    continue
                self.eval_item_counters[iid, col_lp] = np.log1p(
                    sorted_is_lp[s:e][mask].sum()
                )
                self.eval_item_counters[iid, col_like] = np.log1p(
                    sorted_is_like[s:e][mask].sum()
                )
                self.eval_item_counters[iid, col_skip] = np.log1p(
                    sorted_is_skip[s:e][mask].sum()
                )

        logger.info(f"Eval item counters precomputed: {num_items} items")

    def get_user_counters(self, uid: int) -> np.ndarray:
        """Return user counters at end of training (last position)."""
        start, end = self.store.get_user_slice(uid)
        if start < 0 or not self.store.enable_counters:
            W = len(self.config.counter_windows_days)
            return np.zeros(3 * W, dtype=np.float32)
        return self.store.user_counters[end - 1].copy()

    def get_item_counters_batch(self, item_ids: np.ndarray) -> np.ndarray:
        """Return precomputed item counters for a batch of items. [len(item_ids), 3W]"""
        return self.eval_item_counters[item_ids]

    def get_cross_counters_batch(
        self, uid: int, item_ids: np.ndarray,
    ) -> np.ndarray:
        """Compute cross counters for (uid, each candidate item) at eval time. [len(item_ids), 9W]"""
        W = len(self.config.counter_windows_days)
        result = np.zeros((len(item_ids), 9 * W), dtype=np.float32)

        start, end = self.store.get_user_slice(uid)
        if start < 0:
            return result

        max_window = max(self.config.counter_windows_days)
        max_ts = int(self.store.flat_timestamps[end - 1])
        global_window_start = max_ts - max_window * SECONDS_PER_DAY

        ts_slice = self.store.flat_timestamps[start:end]
        first_in_window = int(np.searchsorted(ts_slice, global_window_start))
        window_start_pos = start + first_in_window

        window_items = self.store.flat_item_ids[window_start_pos:end]
        window_ts = self.store.flat_timestamps[window_start_pos:end]
        window_is_lp = self.store.flat_is_listen_plus[window_start_pos:end]
        window_is_like = self.store.flat_is_like[window_start_pos:end]
        window_is_skip = self.store.flat_is_skip[window_start_pos:end]
        window_artists = self.item_to_artist[window_items]
        window_albums = self.item_to_album[window_items]

        for w_idx, w_days in enumerate(self.config.counter_windows_days):
            w_start_ts = max_ts - w_days * SECONDS_PER_DAY
            w_mask = window_ts >= w_start_ts
            w_items = window_items[w_mask]
            w_artists = window_artists[w_mask]
            w_albums = window_albums[w_mask]
            w_lp = window_is_lp[w_mask]
            w_like = window_is_like[w_mask]
            w_skip = window_is_skip[w_mask]
            base_col = w_idx * 9

            for ci, iid in enumerate(item_ids):
                aid = int(self.item_to_artist[iid])
                alid = int(self.item_to_album[iid])

                im = w_items == iid
                am = w_artists == aid
                alm = w_albums == alid

                result[ci, base_col + 0] = np.log1p(w_lp[im].sum())
                result[ci, base_col + 1] = np.log1p(w_like[im].sum())
                result[ci, base_col + 2] = np.log1p(w_skip[im].sum())
                result[ci, base_col + 3] = np.log1p(w_lp[am].sum())
                result[ci, base_col + 4] = np.log1p(w_like[am].sum())
                result[ci, base_col + 5] = np.log1p(w_skip[am].sum())
                result[ci, base_col + 6] = np.log1p(w_lp[alm].sum())
                result[ci, base_col + 7] = np.log1p(w_like[alm].sum())
                result[ci, base_col + 8] = np.log1p(w_skip[alm].sum())

        return result

    def __len__(self) -> int:
        return len(self.test_uids)

    def __getitem__(self, idx: int) -> dict:
        uid = self.test_uids[idx]
        history = self.get_user_history(uid)
        gt = self.test_ground_truth[uid]
        return {
            "uid": uid,
            "history": history,
            "ground_truth_listen_plus": gt["listen_plus"],
            "ground_truth_like": gt["like"],
        }


def collate_to_dict(batch: list[ScoringPair]) -> dict[str, torch.Tensor]:
    """Collate ScoringPairs into batched tensors."""
    result = {
        "hist_lp_item_ids": torch.stack([b.hist_lp_item_ids for b in batch]),
        "hist_lp_artist_ids": torch.stack([b.hist_lp_artist_ids for b in batch]),
        "hist_lp_album_ids": torch.stack([b.hist_lp_album_ids for b in batch]),
        "hist_like_item_ids": torch.stack([b.hist_like_item_ids for b in batch]),
        "hist_like_artist_ids": torch.stack([b.hist_like_artist_ids for b in batch]),
        "hist_like_album_ids": torch.stack([b.hist_like_album_ids for b in batch]),
        "hist_skip_item_ids": torch.stack([b.hist_skip_item_ids for b in batch]),
        "hist_skip_artist_ids": torch.stack([b.hist_skip_artist_ids for b in batch]),
        "hist_skip_album_ids": torch.stack([b.hist_skip_album_ids for b in batch]),
        "uid": torch.tensor([b.uid for b in batch], dtype=torch.long),
        "item_id": torch.tensor([b.item_id for b in batch], dtype=torch.long),
        "artist_id": torch.tensor([b.artist_id for b in batch], dtype=torch.long),
        "album_id": torch.tensor([b.album_id for b in batch], dtype=torch.long),
        "audio_embed": torch.stack([b.audio_embed for b in batch]),
        "listen_plus": torch.tensor([b.listen_plus for b in batch], dtype=torch.float32),
        "like": torch.tensor([b.like for b in batch], dtype=torch.float32),
        "dislike": torch.tensor([b.dislike for b in batch], dtype=torch.float32),
        "listen_pct": torch.tensor([b.listen_pct for b in batch], dtype=torch.float32),
    }
    if batch[0].user_counters is not None:
        result["user_counters"] = torch.stack([b.user_counters for b in batch])
        result["item_counters"] = torch.stack([b.item_counters for b in batch])
        result["cross_counters"] = torch.stack([b.cross_counters for b in batch])
    return result
