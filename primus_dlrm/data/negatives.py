"""Negative sampling strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NegativeSampler(ABC):
    """Base class for negative samplers."""

    def __init__(self, num_items: int, seed: int = 42):
        self.num_items = num_items
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, n: int, exclude: set[int] | None = None) -> np.ndarray:
        """Sample n negative item IDs, optionally excluding some."""
        ...

    def _rejection_sample(self, n: int, exclude: set[int] | None, probs: np.ndarray | None = None) -> np.ndarray:
        if exclude is None or len(exclude) == 0:
            return self.rng.choice(self.num_items, size=n, replace=False, p=probs)

        result = []
        max_attempts = n * 10
        attempts = 0
        while len(result) < n and attempts < max_attempts:
            batch_size = min((n - len(result)) * 3, self.num_items)
            candidates = self.rng.choice(self.num_items, size=batch_size, replace=False, p=probs)
            for c in candidates:
                if c not in exclude:
                    result.append(c)
                    if len(result) >= n:
                        break
            attempts += batch_size
        return np.array(result[:n], dtype=np.int64)


class UniformSampler(NegativeSampler):
    """Sample negatives uniformly at random."""

    def sample(self, n: int, exclude: set[int] | None = None) -> np.ndarray:
        return self._rejection_sample(n, exclude)


class PopularitySampler(NegativeSampler):
    """Sample negatives proportional to item popularity."""

    def __init__(self, num_items: int, item_counts: np.ndarray, seed: int = 42):
        super().__init__(num_items, seed)
        counts = item_counts.astype(np.float64)
        self.probs = counts / counts.sum()

    def sample(self, n: int, exclude: set[int] | None = None) -> np.ndarray:
        return self._rejection_sample(n, exclude, probs=self.probs)


class MixedSampler(NegativeSampler):
    """Mixture of uniform and popularity-weighted sampling."""

    def __init__(
        self,
        num_items: int,
        item_counts: np.ndarray,
        uniform_ratio: float = 0.5,
        seed: int = 42,
    ):
        super().__init__(num_items, seed)
        self.uniform_ratio = uniform_ratio
        self.uniform = UniformSampler(num_items, seed=seed)
        self.popularity = PopularitySampler(num_items, item_counts, seed=seed + 1)

    def sample(self, n: int, exclude: set[int] | None = None) -> np.ndarray:
        n_uniform = int(n * self.uniform_ratio)
        n_pop = n - n_uniform
        u = self.uniform.sample(n_uniform, exclude)
        p = self.popularity.sample(n_pop, exclude)
        result = np.concatenate([u, p])
        self.rng.shuffle(result)
        return result
