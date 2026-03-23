"""Smoke tests for ranking metrics."""
import numpy as np

from primus_dlrm.evaluation.metrics import ndcg_at_k, recall_at_k


def test_ndcg_perfect():
    scores = [(0, 10.0), (1, 9.0), (2, 8.0)]
    gt = {0, 1, 2}
    result = ndcg_at_k(scores, gt, k=3)
    assert abs(result - 1.0) < 1e-6


def test_ndcg_worst():
    scores = [(3, 10.0), (4, 9.0), (0, 8.0)]
    gt = {0}
    result = ndcg_at_k(scores, gt, k=3)
    expected = (1.0 / np.log2(4)) / (1.0 / np.log2(2))
    assert abs(result - expected) < 1e-6


def test_ndcg_empty_gt():
    scores = [(0, 10.0)]
    result = ndcg_at_k(scores, set(), k=3)
    assert result == 0.0


def test_recall_full():
    scores = [(0, 10.0), (1, 9.0), (2, 8.0)]
    gt = {0, 1}
    result = recall_at_k(scores, gt, k=3)
    assert abs(result - 1.0) < 1e-6


def test_recall_partial():
    scores = [(3, 10.0), (4, 9.0), (0, 8.0)]
    gt = {0, 1}
    result = recall_at_k(scores, gt, k=3)
    assert abs(result - 0.5) < 1e-6


def test_recall_empty_gt():
    result = recall_at_k([(0, 1.0)], set(), k=3)
    assert result == 0.0
