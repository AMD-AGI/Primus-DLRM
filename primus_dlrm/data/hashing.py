"""Cross-feature hashing for OneTrans (closed-division reproducible).

xxhash64 of a little-endian byte concat of two int64 keys, modulo the table
size. Bit-reproducible across CPU and GPU architectures (the explicit
``struct.pack("<qq", ...)`` makes byte order independent of host endianness).

Used by the dataset hot path to derive cross-product embedding ids
(``user x artist``, ``user x album``, ``user x hour``, ...) without needing
to materialize them into the on-disk cache.

Reference spec: ``OneTrans_Training_Benchmark_Proposal.md`` Appendix B.
"""
from __future__ import annotations

import struct

import numpy as np
import xxhash


__all__ = ["cross_hash_int64", "cross_hash_batch"]


def cross_hash_int64(
    feat_a: int,
    feat_b: int,
    table_size: int,
    salt: int = 0,
) -> int:
    """xxhash64 of little-endian concat(int64 feat_a, int64 feat_b) mod table_size.

    Args:
        feat_a: first int64 key (e.g. uid).
        feat_b: second int64 key (e.g. artist_id, hour_of_day).
        table_size: hashed embedding table cardinality. Returned id is in
            ``[0, table_size)``.
        salt: optional seed for n-way crosses (e.g. ``user x artist x hour``
            uses one salt for ``(user, artist)`` and a different salt for the
            outer hash to avoid collisions with the 2-way cross). Defaults to
            0 for the standard 2-way crosses.

    Returns:
        The bucket index in ``[0, table_size)``.
    """
    digest = xxhash.xxh64(seed=salt)
    digest.update(struct.pack("<qq", int(feat_a), int(feat_b)))
    return digest.intdigest() % table_size


def cross_hash_batch(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    table_size: int,
    salt: int = 0,
) -> np.ndarray:
    """Vectorised wrapper around ``cross_hash_int64`` over two int64 arrays.

    Used by the eval path (one user, many candidate items) where computing
    one hash per scoring pair would dominate the eval throughput. The hash
    itself is per-pair (xxhash has no SIMD batching), but Python-loop
    overhead is amortized inside C-level numpy iteration.

    Args:
        arr_a: ``[N]`` int64 array of first keys.
        arr_b: ``[N]`` int64 array of second keys (broadcast to ``arr_a``
            length when scalar).
        table_size: hashed table cardinality.
        salt: optional seed (see ``cross_hash_int64``).

    Returns:
        ``[N]`` int64 array of bucket indices.
    """
    arr_a = np.asarray(arr_a, dtype=np.int64).ravel()
    arr_b = np.asarray(arr_b, dtype=np.int64).ravel()
    if arr_b.shape[0] == 1 and arr_a.shape[0] != 1:
        arr_b = np.broadcast_to(arr_b, arr_a.shape)
    elif arr_a.shape[0] == 1 and arr_b.shape[0] != 1:
        arr_a = np.broadcast_to(arr_a, arr_b.shape)
    assert arr_a.shape == arr_b.shape, (
        f"cross_hash_batch shape mismatch: {arr_a.shape} vs {arr_b.shape}"
    )

    out = np.empty(arr_a.shape[0], dtype=np.int64)
    pack = struct.Struct("<qq").pack
    digest_cls = xxhash.xxh64
    for i in range(arr_a.shape[0]):
        d = digest_cls(seed=salt)
        d.update(pack(int(arr_a[i]), int(arr_b[i])))
        out[i] = d.intdigest() % table_size
    return out
