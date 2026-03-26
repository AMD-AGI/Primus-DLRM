#!/usr/bin/env python3
"""Pre-build the FlatEventStore mmap cache for train and eval splits.

Run this once after preprocessing to avoid the cold-start delay when
launching training for the first time.  The cache is written to
``--cache-dir`` (default ``data/cache``) and is picked up automatically
by YambdaTrainDataset / YambdaEvalDataset when ``use_cache=True``.

Examples
--------
# Build all cache variants needed by every config in configs/:
python scripts/build_cache.py --all-configs --processed-dir data/processed

# Single config:
python scripts/build_cache.py --config configs/bench_onetrans_v7_large.yaml

# Manual counter settings:
python scripts/build_cache.py --processed-dir data/processed \
    --enable-counters --counter-windows 7 30
"""
import argparse
import logging
import time

from pathlib import Path

from primus_dlrm.config import Config, DataConfig
from primus_dlrm.data.dataset import (
    FlatEventStore,
    _cache_key,
    _save_store_cache,
    _try_load_cached_store,
)

import numpy as np
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_id_mappings(processed_dir: Path, num_items: int):
    """Load item-to-artist and item-to-album lookup arrays."""
    artist_map = pl.read_parquet(processed_dir / "artist_item_mapping.parquet")
    item_to_artist = np.zeros(num_items, dtype=np.int64)
    valid = artist_map.filter(pl.col("item_id") < num_items)
    item_to_artist[valid["item_id"].to_numpy()] = valid["artist_id"].to_numpy()

    album_map = pl.read_parquet(processed_dir / "album_item_mapping.parquet")
    item_to_album = np.zeros(num_items, dtype=np.int64)
    valid = album_map.filter(pl.col("item_id") < num_items)
    item_to_album[valid["item_id"].to_numpy()] = valid["album_id"].to_numpy()

    return item_to_artist, item_to_album


def build_split_cache(
    config: DataConfig,
    processed_dir: Path,
    split: str,
    sessions: pl.DataFrame | None = None,
    item_to_artist: np.ndarray | None = None,
    item_to_album: np.ndarray | None = None,
) -> None:
    """Build and persist the FlatEventStore cache for one split.

    Accepts pre-loaded sessions / mappings to avoid redundant I/O when
    building multiple cache variants.
    """
    cached = _try_load_cached_store(config, split)
    if cached is not None:
        key = _cache_key(
            split,
            config.counter_windows_days if config.enable_counters else None,
        )
        logger.info(f"[{key}] Cache already exists, skipping.")
        return

    key = _cache_key(
        split,
        config.counter_windows_days if config.enable_counters else None,
    )
    logger.info(f"[{key}] Building FlatEventStore...")
    t0 = time.time()

    if sessions is None:
        sessions = pl.read_parquet(processed_dir / "train_sessions.parquet")

    if config.enable_counters and item_to_artist is None:
        num_items = len(np.load(processed_dir / "item_popularity.npy"))
        item_to_artist, item_to_album = _load_id_mappings(processed_dir, num_items)

    store = FlatEventStore(
        sessions,
        enable_counters=config.enable_counters,
        item_to_artist=item_to_artist if config.enable_counters else None,
        item_to_album=item_to_album if config.enable_counters else None,
        counter_windows_days=config.counter_windows_days if config.enable_counters else None,
    )
    _save_store_cache(store, config, split)

    elapsed = time.time() - t0
    cache_dir = Path(config.cache_dir) / key
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
            dc.enable_counters,
            tuple(dc.counter_windows_days) if dc.enable_counters else (),
        )
        if cache_signature not in seen_keys:
            seen_keys.add(cache_signature)
            configs.append(dc)
            logger.info(
                f"  {yaml_path.name}: counters={dc.enable_counters}, "
                f"windows={dc.counter_windows_days if dc.enable_counters else '(none)'}"
            )

    return configs


def main():
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
        "--processed-dir", type=str, default="data/processed",
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override cache output directory (default: data/cache)",
    )
    parser.add_argument(
        "--enable-counters", action="store_true",
        help="Enable counter feature precomputation",
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

    processed_dir = Path(args.processed_dir)
    if not (processed_dir / "train_sessions.parquet").exists():
        parser.error(
            f"train_sessions.parquet not found in {processed_dir}. "
            f"Run scripts/preprocess.py first."
        )

    # Collect DataConfig variants to build
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

    # Apply cache_dir override to all variants
    for dc in data_configs:
        dc.use_cache = True
        if args.cache_dir is not None:
            dc.cache_dir = args.cache_dir

    # Pre-load shared data once
    logger.info("Loading sessions and metadata...")
    sessions = pl.read_parquet(processed_dir / "train_sessions.parquet")

    need_counters = any(dc.enable_counters for dc in data_configs)
    item_to_artist, item_to_album = None, None
    if need_counters:
        num_items = len(np.load(processed_dir / "item_popularity.npy"))
        item_to_artist, item_to_album = _load_id_mappings(processed_dir, num_items)

    # Build each variant
    t_total = time.time()
    for dc in data_configs:
        for split in args.splits:
            build_split_cache(
                dc, processed_dir, split,
                sessions=sessions,
                item_to_artist=item_to_artist,
                item_to_album=item_to_album,
            )

    # Report total size
    cache_root = Path(data_configs[0].cache_dir)
    if cache_root.exists():
        total_bytes = sum(
            f.stat().st_size for f in cache_root.rglob("*") if f.is_file()
        )
        logger.info(
            f"Total cache size: {total_bytes / 1e9:.2f} GB at {cache_root}"
        )
    logger.info(f"All done in {time.time() - t_total:.1f}s.")


if __name__ == "__main__":
    main()
