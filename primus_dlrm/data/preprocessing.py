"""Preprocess Yambda data: temporal split, session segmentation, metadata."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

EVENT_TYPE_MAP = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}

SECONDS_PER_TIMESTAMP_UNIT = 1  # Yambda timestamps are in seconds (rounded to 5s boundaries)
SECONDS_PER_DAY = 86400
TIMESTAMP_UNITS_PER_DAY = SECONDS_PER_DAY // SECONDS_PER_TIMESTAMP_UNIT


CHUNK_SIZE = 10_000_000  # rows per chunk for large datasets


def preprocess(
    raw_dir: str | Path,
    out_dir: str | Path,
    dataset_size: str = "50m",
    session_gap_seconds: int = 1800,
    train_days: int = 300,
    gap_minutes: int = 30,
    test_days: int = 1,
    chunked: bool = False,
) -> Path:
    """Run full preprocessing pipeline.

    Args:
        chunked: If True, process parquet in chunks (required for 5B dataset).

    Returns:
        Path to the output directory containing processed data.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading multi_event...")
    parquet_path = raw_dir / dataset_size / "multi_event.parquet"

    if chunked:
        events = _load_chunked(parquet_path)
    else:
        events = pl.read_parquet(parquet_path)
    logger.info(f"Loaded {len(events):,} events")

    events = _encode_event_types(events)

    t_min = events["timestamp"].min()
    t_max = events["timestamp"].max()
    logger.info(f"Timestamp range: {t_min} - {t_max} ({(t_max - t_min) / TIMESTAMP_UNITS_PER_DAY:.1f} days)")

    train_start, train_end, test_start, test_end = _compute_split_boundaries(
        t_max, train_days, gap_minutes, test_days
    )
    logger.info(
        f"GTS boundaries: train=[{train_start}, {train_end}), "
        f"gap=[{train_end}, {test_start}), test=[{test_start}, {test_end})"
    )

    train_events, test_events = _apply_temporal_split(events, train_start, train_end, test_start, test_end)
    logger.info(f"Train: {len(train_events):,} events, Test: {len(test_events):,} events")

    session_gap_units = session_gap_seconds // SECONDS_PER_TIMESTAMP_UNIT
    sessions = _build_sessions(train_events, session_gap_units)
    logger.info(f"Built {len(sessions):,} sessions")

    session_index = _build_session_index(sessions)
    logger.info(f"Session index for {len(session_index):,} users")

    item_popularity = _compute_item_popularity(train_events)

    _load_and_save_metadata(raw_dir, out_dir)

    sessions.write_parquet(out_dir / "train_sessions.parquet")
    test_events.write_parquet(out_dir / "test_events.parquet")
    session_index.write_parquet(out_dir / "session_index.parquet")
    np.save(out_dir / "item_popularity.npy", item_popularity)

    split_meta = {
        "t_min": int(t_min),
        "t_max": int(t_max),
        "train_start": int(train_start),
        "train_end": int(train_end),
        "test_start": int(test_start),
        "test_end": int(test_end),
        "train_days": train_days,
        "gap_minutes": gap_minutes,
        "test_days": test_days,
        "session_gap_seconds": session_gap_seconds,
        "num_train_events": len(train_events),
        "num_test_events": len(test_events),
        "num_sessions": len(sessions),
        "num_users": len(session_index),
    }
    import json
    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    logger.info(f"Preprocessing complete. Output: {out_dir}")
    return out_dir


def _encode_event_types(events: pl.DataFrame) -> pl.DataFrame:
    if events["event_type"].dtype == pl.Utf8:
        events = events.with_columns(
            pl.col("event_type").replace_strict(EVENT_TYPE_MAP).cast(pl.UInt8).alias("event_type")
        )
    return events


def _compute_split_boundaries(
    t_max: int, train_days: int, gap_minutes: int, test_days: int
) -> tuple[int, int, int]:
    """GTS: last test_days at the end, then gap, then train_days before that."""
    test_end = t_max
    test_start = test_end - test_days * TIMESTAMP_UNITS_PER_DAY
    gap_units = (gap_minutes * 60) // SECONDS_PER_TIMESTAMP_UNIT
    train_end = test_start - gap_units
    train_start = train_end - train_days * TIMESTAMP_UNITS_PER_DAY
    return train_start, train_end, test_start, test_end


def _apply_temporal_split(
    events: pl.DataFrame, train_start: int, train_end: int,
    test_start: int, test_end: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = events.filter(
        (pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end)
    )
    test_all = events.filter(
        (pl.col("timestamp") >= test_start) & (pl.col("timestamp") < test_end)
    )
    train_users = train.select("uid").unique()
    test = test_all.join(train_users, on="uid", how="inner")
    return train, test


def _build_sessions(events: pl.DataFrame, session_gap_units: int) -> pl.DataFrame:
    """Segment per-user event timelines into sessions based on inactivity gap."""
    sorted_events = events.sort(["uid", "timestamp"])

    sessions = (
        sorted_events
        .with_columns(
            (pl.col("timestamp").diff().over("uid").fill_null(0) > session_gap_units)
            .cum_sum()
            .over("uid")
            .cast(pl.UInt32)
            .alias("session_id")
        )
        .group_by(["uid", "session_id"])
        .agg(
            pl.col("item_id").alias("item_ids"),
            pl.col("timestamp").alias("timestamps"),
            pl.col("event_type").alias("event_types"),
            pl.col("is_organic").alias("is_organic"),
            pl.col("played_ratio_pct").alias("played_ratio_pct"),
            pl.col("track_length_seconds").alias("track_length_seconds"),
        )
        .sort(["uid", "session_id"])
    )
    return sessions


def _build_session_index(sessions: pl.DataFrame) -> pl.DataFrame:
    """Build per-user session index with cumulative event counts."""
    index = (
        sessions
        .with_columns(
            pl.col("item_ids").list.len().alias("session_len")
        )
        .group_by("uid")
        .agg(
            pl.col("session_id").alias("session_ids"),
            pl.col("session_len").alias("session_lens"),
            pl.col("session_len").cum_sum().alias("session_offsets"),
        )
        .sort("uid")
    )
    return index


def _compute_item_popularity(train_events: pl.DataFrame) -> np.ndarray:
    """Compute item frequency counts from training data."""
    counts = (
        train_events
        .group_by("item_id")
        .len()
        .sort("item_id")
    )
    max_item = counts["item_id"].max()
    popularity = np.zeros(max_item + 1, dtype=np.int64)
    popularity[counts["item_id"].to_numpy()] = counts["len"].to_numpy()
    return popularity


def _load_and_save_metadata(raw_dir: Path, out_dir: Path) -> None:
    """Copy/transform metadata files (artist/album mappings, embeddings)."""
    for name in ["artist_item_mapping", "album_item_mapping"]:
        src = raw_dir / f"{name}.parquet"
        dst = out_dir / f"{name}.parquet"
        if src.exists() and not dst.exists():
            df = pl.read_parquet(src)
            df.write_parquet(dst)
            logger.info(f"Copied {name}: {len(df):,} rows")

    emb_src = raw_dir / "embeddings.parquet"
    emb_dst = out_dir / "embeddings.parquet"
    if emb_src.exists() and not emb_dst.exists():
        df = pl.read_parquet(emb_src)
        df.write_parquet(emb_dst)
        logger.info(f"Copied embeddings: {len(df):,} rows")


def _load_chunked(parquet_path: Path) -> pl.DataFrame:
    """Load a large parquet file in chunks using polars scan + streaming."""
    logger.info(f"Chunked loading from {parquet_path}...")
    lf = pl.scan_parquet(parquet_path)
    row_count = lf.select(pl.len()).collect().item()
    logger.info(f"Total rows: {row_count:,}")

    if row_count <= CHUNK_SIZE * 2:
        return pl.read_parquet(parquet_path)

    chunks = []
    for offset in range(0, row_count, CHUNK_SIZE):
        chunk = lf.slice(offset, CHUNK_SIZE).collect()
        chunks.append(chunk)
        logger.info(f"  Loaded chunk {offset:,}-{offset + len(chunk):,}")

    return pl.concat(chunks)
