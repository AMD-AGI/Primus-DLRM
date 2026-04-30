"""Numba-parallelised replacements for FlatEventStore counter precompute.

Bit-exact replicas of the original two-pointer Python loops in
``primus_dlrm/data/dataset.py`` (``_precompute_user_counters``,
``_precompute_item_counters``, ``_precompute_cross_counters``) but compiled
with numba and parallelised over independent segments (per user, per item,
per (user, entity) pair). On a 128-core box this runs roughly 100--500x
faster than the original CPython implementation while producing identical
output (counts and ``log1p`` values match to floating-point bitwise).

Why bit-exact rather than vectorised polars?
--------------------------------------------
The original code's two-pointer loop has a subtle interaction with tied
timestamps and the broader "skip-bucket" definition in cross counters
(``evt = 0 if is_lp else (1 if is_like else 2)``) that is impossible to
replicate with polars' ``rolling_sum_by`` without per-row tiebreakers and
boundary handling. Numba sidesteps this by translating the original Python
loop directly into a tight native loop -- same control flow, same arithmetic.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
import numba as nb
import polars as pl

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400


# ---------------------------------------------------------------------------
# Segment-level kernels (all bit-exact replicas of the original Python loops)
# ---------------------------------------------------------------------------


@nb.njit(parallel=True, cache=True, boundscheck=False)
def _user_counters_kernel(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    timestamps: np.ndarray,
    is_lp: np.ndarray,
    is_like: np.ndarray,
    is_skip: np.ndarray,
    window_sec: np.int64,
    out: np.ndarray,
    col_lp: np.int64,
    col_like: np.int64,
    col_skip: np.int64,
) -> None:
    """For each segment [s, e), apply the two-pointer sliding-window counter.

    Mirrors the inner per-user loop of ``_precompute_user_counters``.
    """
    n_seg = seg_starts.shape[0]
    for seg in nb.prange(n_seg):
        s = seg_starts[seg]
        e = seg_ends[seg]
        left = s
        cum_lp = np.int64(0)
        cum_like = np.int64(0)
        cum_skip = np.int64(0)
        for pos in range(s, e):
            window_start_ts = timestamps[pos] - window_sec
            while left < pos and timestamps[left] < window_start_ts:
                cum_lp -= np.int64(is_lp[left])
                cum_like -= np.int64(is_like[left])
                cum_skip -= np.int64(is_skip[left])
                left += 1
            out[pos, col_lp] = np.log1p(np.float64(cum_lp))
            out[pos, col_like] = np.log1p(np.float64(cum_like))
            out[pos, col_skip] = np.log1p(np.float64(cum_skip))
            cum_lp += np.int64(is_lp[pos])
            cum_like += np.int64(is_like[pos])
            cum_skip += np.int64(is_skip[pos])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_segments(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, ends)`` int64 arrays such that within each [s, e)
    range the values of ``group_ids`` are constant.

    ``group_ids`` must be non-decreasing within segments (i.e. already
    grouped). Equivalent to ``itertools.groupby`` but vectorised.
    """
    n = group_ids.shape[0]
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    changes = np.where(np.diff(group_ids) != 0)[0] + 1
    starts = np.concatenate([[0], changes]).astype(np.int64, copy=False)
    ends = np.concatenate([changes, [n]]).astype(np.int64, copy=False)
    return starts, ends


def _compute_segments_2col(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Segment boundaries for a composite key (a, b).

    Both arrays must already be sorted such that consecutive equal-(a, b)
    rows are adjacent.
    """
    n = a.shape[0]
    if n == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    diff_a = np.diff(a) != 0
    diff_b = np.diff(b) != 0
    changes = np.where(diff_a | diff_b)[0] + 1
    starts = np.concatenate([[0], changes]).astype(np.int64, copy=False)
    ends = np.concatenate([changes, [n]]).astype(np.int64, copy=False)
    return starts, ends


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def precompute_user_counters(
    flat_uid: np.ndarray,
    flat_ts: np.ndarray,
    flat_is_lp: np.ndarray,
    flat_is_like: np.ndarray,
    flat_is_skip: np.ndarray,
    counter_windows_days: Iterable[int],
) -> np.ndarray:
    """Bit-exact, multi-threaded replacement for
    ``FlatEventStore._precompute_user_counters``.

    ``flat_*`` arrays must already be sorted by (uid, ts), as produced by
    ``FlatEventStore.__init__``.
    """
    counter_windows_days = list(counter_windows_days)
    N = flat_uid.shape[0]
    W = len(counter_windows_days)
    out = np.zeros((N, 3 * W), dtype=np.float32)

    seg_s, seg_e = _compute_segments(flat_uid)
    timestamps = flat_ts.astype(np.int64, copy=False)
    is_lp_u8 = flat_is_lp.astype(np.uint8, copy=False)
    is_like_u8 = flat_is_like.astype(np.uint8, copy=False)
    is_skip_u8 = flat_is_skip.astype(np.uint8, copy=False)

    out64 = np.zeros((N, 3 * W), dtype=np.float64)

    for w_idx, w_days in enumerate(counter_windows_days):
        t0 = time.time()
        _user_counters_kernel(
            seg_s, seg_e,
            timestamps, is_lp_u8, is_like_u8, is_skip_u8,
            np.int64(w_days * SECONDS_PER_DAY),
            out64,
            np.int64(w_idx * 3 + 0),
            np.int64(w_idx * 3 + 1),
            np.int64(w_idx * 3 + 2),
        )
        logger.info(
            f"  user_counters[{w_days}d]: {time.time() - t0:.1f}s "
            f"({len(seg_s):,} segments, {N:,} events)"
        )
    out[:] = out64.astype(np.float32, copy=False)
    return out


def _polars_lexsort(
    keys: dict[str, np.ndarray],
) -> np.ndarray:
    """Stable multi-threaded lexicographic sort, returning the permutation
    that orders rows by the given keys in declaration order (most-significant
    first, e.g. ``keys = {'a': ..., 'b': ...}`` sorts primarily by ``a`` then
    by ``b``).

    Stability matters here: the numba two-pointer kernel is order-sensitive
    within a segment when multiple events share the same timestamp (events
    appearing earlier in the sorted array are counted as "before" later ones).
    The original ``np.lexsort`` is stable, so we make the polars sort behave
    identically by appending the original row index as the final sort key.
    Polars' multi-threaded radix sort is typically 5-15x faster than
    np.lexsort on multi-million-row int arrays.
    """
    n = next(iter(keys.values())).shape[0]
    df = pl.DataFrame({"_idx": np.arange(n, dtype=np.int64), **keys})
    return (
        df.sort(by=list(keys.keys()) + ["_idx"], maintain_order=False)
        ["_idx"]
        .to_numpy()
    )


def precompute_item_counters(
    flat_item_ids: np.ndarray,
    flat_ts: np.ndarray,
    flat_is_lp: np.ndarray,
    flat_is_like: np.ndarray,
    flat_is_skip: np.ndarray,
    counter_windows_days: Iterable[int],
) -> np.ndarray:
    """Bit-exact, multi-threaded replacement for
    ``FlatEventStore._precompute_item_counters``.

    Sorts events by (item_id, ts), runs the per-item two-pointer kernel in
    parallel, then scatters results back to original event order.
    """
    counter_windows_days = list(counter_windows_days)
    N = flat_item_ids.shape[0]
    W = len(counter_windows_days)
    out = np.zeros((N, 3 * W), dtype=np.float32)

    t0 = time.time()
    sort_idx = _polars_lexsort({"item_id": flat_item_ids, "ts": flat_ts})
    logger.info(f"  item_counters: polars sort in {time.time() - t0:.1f}s")
    sorted_items = flat_item_ids[sort_idx]
    sorted_ts = flat_ts[sort_idx].astype(np.int64, copy=False)
    sorted_lp = flat_is_lp[sort_idx].astype(np.uint8, copy=False)
    sorted_like = flat_is_like[sort_idx].astype(np.uint8, copy=False)
    sorted_skip = flat_is_skip[sort_idx].astype(np.uint8, copy=False)

    seg_s, seg_e = _compute_segments(sorted_items)

    sorted_out = np.zeros((N, 3 * W), dtype=np.float64)
    for w_idx, w_days in enumerate(counter_windows_days):
        t0 = time.time()
        _user_counters_kernel(
            seg_s, seg_e,
            sorted_ts, sorted_lp, sorted_like, sorted_skip,
            np.int64(w_days * SECONDS_PER_DAY),
            sorted_out,
            np.int64(w_idx * 3 + 0),
            np.int64(w_idx * 3 + 1),
            np.int64(w_idx * 3 + 2),
        )
        logger.info(
            f"  item_counters[{w_days}d]: {time.time() - t0:.1f}s "
            f"({len(seg_s):,} segments, {N:,} events)"
        )
    out[sort_idx] = sorted_out.astype(np.float32, copy=False)
    return out


def _cross_counter_one_entity(
    flat_uid: np.ndarray,
    entity_arr: np.ndarray,
    flat_ts_i64: np.ndarray,
    is_lp_u8: np.ndarray,
    is_like_u8: np.ndarray,
    is_skip_broad_u8: np.ndarray,
    counter_windows_days: list[int],
    entity_name: str,
    entity_idx: int,
    out: np.ndarray,
    user_starts: np.ndarray,
    user_ends: np.ndarray,
    chunk_size: int,
) -> None:
    """Process one entity (item/artist/album) in user-chunks to limit memory."""
    N = flat_uid.shape[0]
    W = len(counter_windows_days)
    n_users = len(user_starts)
    n_chunks = (n_users + chunk_size - 1) // chunk_size

    logger.info(
        f"  cross[{entity_name}]: processing {n_users:,} users in "
        f"{n_chunks} chunks of {chunk_size:,}"
    )
    t_total = time.time()

    for chunk_i in range(n_chunks):
        u_lo = chunk_i * chunk_size
        u_hi = min(u_lo + chunk_size, n_users)
        ev_lo = int(user_starts[u_lo])
        ev_hi = int(user_ends[u_hi - 1])
        chunk_n = ev_hi - ev_lo

        t0 = time.time()
        c_uid = flat_uid[ev_lo:ev_hi]
        c_ent = entity_arr[ev_lo:ev_hi]
        c_ts = flat_ts_i64[ev_lo:ev_hi]
        c_lp = is_lp_u8[ev_lo:ev_hi]
        c_like = is_like_u8[ev_lo:ev_hi]
        c_skip = is_skip_broad_u8[ev_lo:ev_hi]

        sort_idx = _polars_lexsort({
            "uid": c_uid, "ent": c_ent, "ts": c_ts,
        })
        sorted_uid = c_uid[sort_idx]
        sorted_ent = c_ent[sort_idx]
        sorted_ts = c_ts[sort_idx]
        sorted_lp = c_lp[sort_idx]
        sorted_like = c_like[sort_idx]
        sorted_skip = c_skip[sort_idx]
        seg_s, seg_e = _compute_segments_2col(sorted_uid, sorted_ent)

        sorted_out = np.zeros((chunk_n, 3 * W), dtype=np.float64)
        for w_idx, w_days in enumerate(counter_windows_days):
            _user_counters_kernel(
                seg_s, seg_e,
                sorted_ts, sorted_lp, sorted_like, sorted_skip,
                np.int64(w_days * SECONDS_PER_DAY),
                sorted_out,
                np.int64(w_idx * 3 + 0),
                np.int64(w_idx * 3 + 1),
                np.int64(w_idx * 3 + 2),
            )

        global_idx = sort_idx + ev_lo
        for w_idx in range(W):
            base = w_idx * 9 + entity_idx * 3
            out[global_idx, base + 0] = sorted_out[:, w_idx * 3 + 0].astype(
                np.float32, copy=False
            )
            out[global_idx, base + 1] = sorted_out[:, w_idx * 3 + 1].astype(
                np.float32, copy=False
            )
            out[global_idx, base + 2] = sorted_out[:, w_idx * 3 + 2].astype(
                np.float32, copy=False
            )

        del sort_idx, sorted_uid, sorted_ent, sorted_ts
        del sorted_lp, sorted_like, sorted_skip, sorted_out

        elapsed_chunk = time.time() - t0
        elapsed_total = time.time() - t_total
        pct = (chunk_i + 1) / n_chunks * 100
        events_done = ev_hi
        events_per_sec = events_done / max(elapsed_total, 0.01)
        eta = (N - events_done) / max(events_per_sec, 1)
        if (chunk_i + 1) % max(1, n_chunks // 20) == 0 or chunk_i == n_chunks - 1:
            logger.info(
                f"  cross[{entity_name}]: {pct:5.1f}% chunk {chunk_i+1}/{n_chunks} "
                f"| {chunk_n:,} events in {elapsed_chunk:.1f}s "
                f"| {len(seg_s):,} segments "
                f"| elapsed {elapsed_total:.0f}s, ETA {eta:.0f}s"
            )

    logger.info(
        f"  cross[{entity_name}]: done in {time.time() - t_total:.1f}s"
    )


def precompute_cross_counters(
    flat_uid: np.ndarray,
    flat_item_ids: np.ndarray,
    flat_ts: np.ndarray,
    flat_is_lp: np.ndarray,
    flat_is_like: np.ndarray,
    flat_is_skip: np.ndarray,
    item_to_artist: np.ndarray,
    item_to_album: np.ndarray,
    counter_windows_days: Iterable[int],
    chunk_users: int = 100_000,
) -> np.ndarray:
    """Bit-exact, multi-threaded cross counter computation.

    Processes users in chunks to limit peak memory. Each chunk sorts and
    computes counters for a subset of users independently (cross counters
    are per-(user, entity) so chunks are independent).

    Args:
        chunk_users: number of users per chunk. Lower = less memory.
            Default 100k users keeps peak sort memory under ~50 GB for
            typical event distributions. The all-at-once alternative
            allocates >300 GB transient on 5B-scale data (4.7B events,
            1M users) and OOMs if CPU memory is exhausted. Output is bit-identical regardless of
            ``chunk_users``.
    """
    counter_windows_days = list(counter_windows_days)
    N = flat_uid.shape[0]
    W = len(counter_windows_days)
    out = np.zeros((N, 9 * W), dtype=np.float32)

    is_lp_u8 = flat_is_lp.astype(np.uint8, copy=False)
    is_like_u8 = flat_is_like.astype(np.uint8, copy=False)
    is_skip_broad_u8 = (
        (1 - is_lp_u8.astype(np.int32)) * (1 - is_like_u8.astype(np.int32))
    ).astype(np.uint8)

    flat_ts_i64 = flat_ts.astype(np.int64, copy=False)
    flat_artist = item_to_artist[flat_item_ids]
    flat_album = item_to_album[flat_item_ids]

    user_starts, user_ends = _compute_segments(flat_uid)
    logger.info(
        f"  cross counters: {len(user_starts):,} users, {N:,} events, "
        f"chunk_users={chunk_users:,}"
    )

    for entity_idx, (entity_name, entity_arr) in enumerate(
        [("item", flat_item_ids), ("artist", flat_artist), ("album", flat_album)]
    ):
        _cross_counter_one_entity(
            flat_uid, entity_arr, flat_ts_i64,
            is_lp_u8, is_like_u8, is_skip_broad_u8,
            counter_windows_days, entity_name, entity_idx,
            out, user_starts, user_ends, chunk_users,
        )

    return out
