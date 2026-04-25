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
) -> np.ndarray:
    """Bit-exact, multi-threaded replacement for
    ``FlatEventStore._precompute_cross_counters``.

    The original Python implementation iterates per user and uses three
    parallel ``defaultdict`` to track per-item / per-artist / per-album
    counts within the user's sliding window. Equivalently, we can split the
    work into three independent passes:

      pass A: for each event, count user's prior events (in window) with the
              SAME item as this event's target  -> sort by (uid, item_id, ts),
              two-pointer per (uid, item_id) segment.
      pass B: same, for artist  -> sort by (uid, artist_id, ts),
              two-pointer per (uid, artist_id) segment.
      pass C: same, for album  -> sort by (uid, album_id, ts),
              two-pointer per (uid, album_id) segment.

    Each pass can be parallelised across segments. Because all three passes
    use the same per-segment two-pointer logic with the same ``is_lp``,
    ``is_like``, broader-``is_skip`` indicator buckets, this matches the
    original Python output bit-for-bit.

    NOTE: the original ``_precompute_cross_counters`` uses ``evt = 0 if is_lp
    else (1 if is_like else 2)``, putting EVERY event that is not a
    listen-plus and not a like into the "skip" bucket -- including unlikes,
    dislikes, and undislikes. This is broader than the standalone ``is_skip``
    flag (which is only listens with played < 50%). We replicate this
    semantics by passing ``is_skip_broad = ~(is_lp | is_like)`` instead of
    the per-event ``is_skip`` flag. The user_counters and item_counters
    paths use the proper ``is_skip`` indicator, matching the original.
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

    for entity_idx, (entity_name, entity_arr) in enumerate(
        [("item", flat_item_ids), ("artist", flat_artist), ("album", flat_album)]
    ):
        t0 = time.time()
        # Sort by (uid, entity_id, ts) so that each (uid, entity_id) pair forms
        # a single contiguous segment. polars sort is multi-threaded.
        sort_idx = _polars_lexsort({
            "uid": flat_uid, "ent": entity_arr, "ts": flat_ts,
        })
        sorted_uid = flat_uid[sort_idx]
        sorted_ent = entity_arr[sort_idx]
        sorted_ts = flat_ts_i64[sort_idx]
        sorted_lp = is_lp_u8[sort_idx]
        sorted_like = is_like_u8[sort_idx]
        sorted_skip = is_skip_broad_u8[sort_idx]
        seg_s, seg_e = _compute_segments_2col(sorted_uid, sorted_ent)
        logger.info(
            f"  cross[{entity_name}]: sort+segment in {time.time() - t0:.1f}s "
            f"({len(seg_s):,} unique (uid, {entity_name}) segments)"
        )

        sorted_out = np.zeros((N, 3 * W), dtype=np.float64)
        for w_idx, w_days in enumerate(counter_windows_days):
            t1 = time.time()
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
                f"  cross[{entity_name},{w_days}d]: {time.time() - t1:.1f}s"
            )

        # Scatter back into the (entity-block) columns of the output.
        # Original's per-window column layout for cross is:
        #   base = w_idx * 9; then [item_lp, item_like, item_skip,
        #                           artist_lp, artist_like, artist_skip,
        #                           album_lp,  album_like,  album_skip]
        for w_idx in range(W):
            base = w_idx * 9 + entity_idx * 3
            out[sort_idx, base + 0] = sorted_out[:, w_idx * 3 + 0].astype(
                np.float32, copy=False
            )
            out[sort_idx, base + 1] = sorted_out[:, w_idx * 3 + 1].astype(
                np.float32, copy=False
            )
            out[sort_idx, base + 2] = sorted_out[:, w_idx * 3 + 2].astype(
                np.float32, copy=False
            )

    return out
