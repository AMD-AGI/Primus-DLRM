"""Smoke tests for negative samplers."""
import numpy as np

from primus_dlrm.data.negatives import UniformSampler, PopularitySampler, MixedSampler


def test_uniform_sampler():
    sampler = UniformSampler(1000, seed=42)
    negs = sampler.sample(10)
    assert len(negs) == 10
    assert len(set(negs)) == 10
    assert all(0 <= n < 1000 for n in negs)


def test_uniform_exclusion():
    sampler = UniformSampler(100, seed=42)
    exclude = {0, 1, 2, 3, 4}
    negs = sampler.sample(5, exclude=exclude)
    assert len(negs) == 5
    assert not (set(negs) & exclude)


def test_popularity_sampler():
    counts = np.ones(100, dtype=np.int64)
    counts[0] = 1000  # item 0 is very popular
    sampler = PopularitySampler(100, counts, seed=42)
    negs = sampler.sample(50)
    assert len(negs) == 50
    assert 0 in negs  # very likely


def test_mixed_sampler():
    counts = np.ones(100, dtype=np.int64)
    sampler = MixedSampler(100, counts, uniform_ratio=0.5, seed=42)
    negs = sampler.sample(10)
    assert len(negs) == 10
