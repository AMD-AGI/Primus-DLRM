#!/usr/bin/env python3
"""Pre-build the FlatEventStore mmap cache for train and eval splits.

Run this once after preprocessing to avoid the cold-start delay when
launching training for the first time. The cache is written to the
``DataPaths.cache`` location derived from ``--data-root`` and the config's
``dataset_size`` (e.g. ``data/cache_5b/`` for the 5B variant), and is picked
up automatically by ``YambdaTrainDataset`` / ``YambdaEvalDataset`` when
``use_cache=True``.

Examples
--------
# Build all cache variants needed by every config in configs/:
python scripts/build_cache.py --all-configs

# Single config (uses the config's dataset_size to derive paths):
python scripts/build_cache.py --config configs/bench_onetrans_large_5b.yaml

# Manual counter settings against the 50m default layout:
python scripts/build_cache.py --enable-counters --counter-windows 7 30
"""
import argparse
import logging
import time

from pathlib import Path

from primus_dlrm.config import Config, DataConfig
from primus_dlrm.data.dataset import (
    DataPaths,
    FlatEventStore,
    _cache_key,
    _save_store_cache,
    _try_load_cached_store,
)

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_id_mappings(metadata_dir: Path, num_items: int):
    """Load item-to-artist and item-to-album lookup arrays from the
    canonical metadata dir (typically ``data/shared_metadata/``)."""
    artist_map = pl.read_parquet(metadata_dir / "artist_item_mapping.parquet")
    item_to_artist = np.zeros(num_items, dtype=np.int64)
    valid = artist_map.filter(pl.col("item_id") < num_items)
    item_to_artist[valid["item_id"].to_numpy()] = valid["artist_id"].to_numpy()

    album_map = pl.read_parquet(metadata_dir / "album_item_mapping.parquet")
    item_to_album = np.zeros(num_items, dtype=np.int64)
    valid = album_map.filter(pl.col("item_id") < num_items)
    item_to_album[valid["item_id"].to_numpy()] = valid["album_id"].to_numpy()

    return item_to_artist, item_to_album


def build_split_cache(
    config: DataConfig,
    paths: DataPaths,
    split: str,
    sessions: pl.DataFrame | None = None,
    item_to_artist: np.ndarray | None = None,
    item_to_album: np.ndarray | None = None,
) -> None:
    """Build and persist the FlatEventStore cache for one split.

    Accepts pre-loaded sessions / mappings to avoid redundant I/O when
    building multiple cache variants.
    """
    cached = _try_load_cached_store(config, paths, split)
    key = _cache_key(
        split, config.counter_windows_days if config.enable_counters else None,
    )
    t0 = time.time()
    if cached is not None:
        store = cached
        logger.info(f"[{key}] Store cache exists at {paths.cache / key}.")
    else:
        logger.info(f"[{key}] Building FlatEventStore...")

        if sessions is None:
            sessions = pl.read_parquet(paths.processed / "train_sessions.parquet")

        if config.enable_counters and item_to_artist is None:
            num_items = len(np.load(paths.processed / "item_popularity.npy"))
            item_to_artist, item_to_album = _load_id_mappings(paths.metadata, num_items)

        store = FlatEventStore(
            sessions,
            enable_counters=config.enable_counters,
            item_to_artist=item_to_artist if config.enable_counters else None,
            item_to_album=item_to_album if config.enable_counters else None,
            counter_windows_days=config.counter_windows_days if config.enable_counters else None,
        )
        _save_store_cache(store, config, paths, split)

    # Pre-build positions arrays for common history_length values
    cache_dir = paths.cache / key
    for hl in [config.history_length]:
        pos_path = cache_dir / f"positions_L{hl}.npy"
        if pos_path.exists():
            logger.info(f"[{key}] positions_L{hl} already cached.")
            continue
        t_pos = time.time()
        starts = store.user_start
        ends = store.user_end
        counts = np.maximum(ends - starts - hl, 0)
        total = int(counts.sum())
        positions = np.empty(total, dtype=np.int64)
        offset = 0
        for i in range(len(starts)):
            n = int(counts[i])
            if n > 0:
                positions[offset:offset + n] = np.arange(
                    int(starts[i]) + hl, int(ends[i]), dtype=np.int64,
                )
                offset += n
        np.save(pos_path, positions)
        logger.info(
            f"[{key}] positions_L{hl}: {total:,} positions, "
            f"{positions.nbytes / 1e9:.2f} GB, {time.time() - t_pos:.1f}s"
        )

    elapsed = time.time() - t0
    total_bytes = sum(f.stat().st_size for f in cache_dir.iterdir())
    logger.info(
        f"[{key}] Cache built in {elapsed:.1f}s — "
        f"{total_bytes / 1e9:.2f} GB at {cache_dir}"
    )


def _collect_data_configs(config_dir: Path) -> list[DataConfig]:
    """Extract unique DataConfig variants from all YAML configs."""
    seen_keys: set[tuple] = set()
    configs: list[DataConfig] = []

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = Config.load(yaml_path)
        except Exception as e:
            logger.warning(f"Skipping {yaml_path.name}: {e}")
            continue

        dc = cfg.data
        cache_signature = (
            dc.dataset_size,
            dc.enable_counters,
            tuple(dc.counter_windows_days) if dc.enable_counters else (),
        )
        if cache_signature not in seen_keys:
            seen_keys.add(cache_signature)
            configs.append(dc)
            logger.info(
                f"  {yaml_path.name}: size={dc.dataset_size}, "
                f"counters={dc.enable_counters}, "
                f"windows={dc.counter_windows_days if dc.enable_counters else '(none)'}"
            )

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build FlatEventStore mmap cache",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--config", type=str, default=None,
        help="Single YAML config file (uses data section)",
    )
    source.add_argument(
        "--all-configs", action="store_true",
        help="Scan configs/ dir and build caches for every unique variant",
    )
    parser.add_argument(
        "--config-dir", type=str, default="configs",
        help="Directory to scan with --all-configs (default: configs/)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for raw/processed/cache subdirs. Overrides "
             "config.data.data_dir (default: from config, typically 'data').",
    )
    parser.add_argument(
        "--dataset-size", type=str, default=None, choices=["50m", "500m", "5b"],
        help="Override dataset size (default: from config.data.dataset_size).",
    )
    parser.add_argument(
        "--enable-counters", action="store_true",
        help="Enable counter feature precomputation (only used without --config)",
    )
    parser.add_argument(
        "--counter-windows", type=int, nargs="+", default=None,
        help="Counter window sizes in days, e.g. --counter-windows 7 30",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "eval"],
        choices=["train", "eval"],
        help="Which splits to build cache for (default: both)",
    )
    args = parser.parse_args()

    # Resolve DataConfig variants and build a DataPaths for each.
    if args.all_configs:
        config_dir = Path(args.config_dir)
        logger.info(f"Scanning {config_dir} for unique cache variants...")
        data_configs = _collect_data_configs(config_dir)
        if not data_configs:
            parser.error(f"No valid YAML configs found in {config_dir}")
    elif args.config:
        data_configs = [Config.load(args.config).data]
    else:
        dc = DataConfig()
        if args.enable_counters:
            dc.enable_counters = True
        if args.counter_windows is not None:
            dc.counter_windows_days = args.counter_windows
        data_configs = [dc]

    # Apply CLI overrides.
    for dc in data_configs:
        dc.use_cache = True
        if args.dataset_size is not None:
            dc.dataset_size = args.dataset_size

    # Group configs by their DataPaths so we share session+metadata loads.
    by_paths: dict[tuple[str, str], list[DataConfig]] = {}
    for dc in data_configs:
        paths = DataPaths.from_config(dc, data_root=args.data_root)
        by_paths.setdefault((str(paths.data_root), paths.dataset_size), []).append(dc)

    t_total = time.time()
    for (root, size), dcs in by_paths.items():
        paths = DataPaths(data_root=Path(root), dataset_size=size)
        if not (paths.processed / "train_sessions.parquet").exists():
            parser.error(
                f"train_sessions.parquet not found in {paths.processed}. "
                f"Run scripts/preprocess.py --size {size} first."
            )

        logger.info(
            f"=== {size}: processed={paths.processed}, metadata={paths.metadata}, "
            f"cache={paths.cache} ==="
        )

        sessions = pl.read_parquet(paths.processed / "train_sessions.parquet")

        need_counters = any(dc.enable_counters for dc in dcs)
        item_to_artist, item_to_album = None, None
        if need_counters:
            num_items = len(np.load(paths.processed / "item_popularity.npy"))
            item_to_artist, item_to_album = _load_id_mappings(paths.metadata, num_items)

        for dc in dcs:
            for split in args.splits:
                build_split_cache(
                    dc, paths, split,
                    sessions=sessions,
                    item_to_artist=item_to_artist,
                    item_to_album=item_to_album,
                )

        if paths.cache.exists():
            total_bytes = sum(
                f.stat().st_size for f in paths.cache.rglob("*") if f.is_file()
            )
            logger.info(
                f"=== {size}: total cache size {total_bytes / 1e9:.2f} GB "
                f"at {paths.cache} ==="
            )

    logger.info(f"All done in {time.time() - t_total:.1f}s.")


if __name__ == "__main__":
    main()
