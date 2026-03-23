"""Ranking metrics: NDCG@k, Recall@k."""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def ndcg_at_k(scores: np.ndarray, ground_truth: set[int], k: int) -> float:
    """Compute NDCG@k.

    Args:
        scores: Array of (item_id, score) sorted by score descending, or
                a dict mapping item_id -> score.
        ground_truth: Set of relevant item IDs.
        k: Cutoff.

    Returns:
        NDCG@k value.
    """
    if len(ground_truth) == 0:
        return 0.0

    if isinstance(scores, dict):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    else:
        ranked = scores

    dcg = 0.0
    for i, (item_id, _score) in enumerate(ranked[:k]):
        if item_id in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    ideal_len = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(scores: np.ndarray | dict, ground_truth: set[int], k: int) -> float:
    """Compute Recall@k."""
    if len(ground_truth) == 0:
        return 0.0

    if isinstance(scores, dict):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    else:
        ranked = scores

    top_k_items = {item_id for item_id, _ in ranked[:k]}
    hits = len(top_k_items & ground_truth)
    return hits / len(ground_truth)


def evaluate_ranking(
    model: torch.nn.Module,
    eval_dataset,
    device: torch.device,
    candidate_item_ids: np.ndarray | None = None,
    ks: list[int] | None = None,
    task: str = "listen_plus",
    batch_size: int = 1024,
    log_interval: int = 0,
) -> dict[str, float]:
    """Evaluate ranking metrics for all test users.

    Args:
        model: Trained model.
        eval_dataset: YambdaEvalDataset instance.
        device: Device to run on.
        candidate_item_ids: Item IDs to score against. If None, uses top-N popular items.
        ks: List of k values for metrics. Default [10, 100].
        task: Which task head to use for ranking. Default "listen_plus".
        batch_size: Batch size for scoring candidates.
        log_interval: Log progress every N users. 0 = no progress logging.

    Returns:
        Dict of metric_name -> value, averaged across users.
    """
    if ks is None:
        ks = [10, 100]

    model.eval()
    all_ndcg = {k: [] for k in ks}
    all_recall = {k: [] for k in ks}

    if candidate_item_ids is None:
        pop = eval_dataset.item_popularity
        candidate_item_ids = np.argsort(-pop)[:10000]

    total_users = len(eval_dataset)
    with torch.no_grad():
        for idx in range(total_users):
            sample = eval_dataset[idx]
            uid = sample["uid"]
            history = sample["history"]

            gt_key = f"ground_truth_{task}"
            ground_truth = sample.get(gt_key, set())
            if len(ground_truth) == 0:
                continue

            user_scores = _score_candidates(
                model, history, uid, candidate_item_ids,
                eval_dataset, device, task, batch_size,
            )

            for k in ks:
                all_ndcg[k].append(ndcg_at_k(user_scores, ground_truth, k))
                all_recall[k].append(recall_at_k(user_scores, ground_truth, k))

            if log_interval > 0 and (idx + 1) % log_interval == 0:
                logger.info(f"  [global] {idx + 1}/{total_users} users evaluated")

    results = {}
    for k in ks:
        results[f"ndcg@{k}"] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        results[f"recall@{k}"] = np.mean(all_recall[k]) if all_recall[k] else 0.0
    results["num_users"] = len(all_ndcg[ks[0]])
    return results


def evaluate_ranking_peruser(
    model: torch.nn.Module,
    eval_dataset,
    device: torch.device,
    ks: list[int] | None = None,
    task: str = "listen_plus",
    top_n: int = 100,
    batch_size: int = 1024,
    log_interval: int = 0,
) -> dict[str, float]:
    """Evaluate ranking with per-user candidate pools.

    For each test user, the candidate pool is all unique items they interacted
    with during training. The model scores all of them, and NDCG/Recall are
    computed on the top-N by score.

    Args:
        model: Trained model.
        eval_dataset: YambdaEvalDataset instance (must have get_user_train_items).
        device: Device to run on.
        ks: List of k values for metrics. Default [10, 50, 100].
        task: Which task head to use for ranking.
        top_n: Number of top-scored items to keep for metric computation.
        batch_size: Batch size for scoring candidates.
        log_interval: Log progress every N users. 0 = no progress logging.

    Returns:
        Dict of metric_name -> value, averaged across users.
    """
    if ks is None:
        ks = [10, 50, 100]

    model.eval()
    all_ndcg = {k: [] for k in ks}
    all_recall = {k: [] for k in ks}

    total_users = len(eval_dataset)
    with torch.no_grad():
        for idx in range(total_users):
            sample = eval_dataset[idx]
            uid = sample["uid"]
            history = sample["history"]

            gt_key = f"ground_truth_{task}"
            ground_truth = sample.get(gt_key, set())
            if len(ground_truth) == 0:
                continue

            candidate_item_ids = eval_dataset.get_user_train_items(uid)
            if len(candidate_item_ids) == 0:
                continue

            user_scores = _score_candidates(
                model, history, uid, candidate_item_ids,
                eval_dataset, device, task, batch_size,
            )

            user_scores = user_scores[:top_n]

            for k in ks:
                all_ndcg[k].append(ndcg_at_k(user_scores, ground_truth, k))
                all_recall[k].append(recall_at_k(user_scores, ground_truth, k))

            if log_interval > 0 and (idx + 1) % log_interval == 0:
                logger.info(f"  [peruser] {idx + 1}/{total_users} users evaluated")

    results = {}
    for k in ks:
        results[f"ndcg@{k}"] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        results[f"recall@{k}"] = np.mean(all_recall[k]) if all_recall[k] else 0.0
    results["num_users"] = len(all_ndcg[ks[0]])
    return results


def _score_candidates(
    model: torch.nn.Module,
    history: dict,
    uid: int,
    candidate_item_ids: np.ndarray,
    eval_dataset,
    device: torch.device,
    task: str,
    batch_size: int,
) -> list[tuple[int, float]]:
    """Score all candidate items for one user."""
    scores = []
    num_candidates = len(candidate_item_ids)
    enable_counters = getattr(eval_dataset, "config", None) and eval_dataset.config.enable_counters

    hist_keys = [
        "lp_items", "lp_artists", "lp_albums",
        "like_items", "like_artists", "like_albums",
        "skip_items", "skip_artists", "skip_albums",
    ]
    batch_key_map = {
        "lp_items": "hist_lp_item_ids",
        "lp_artists": "hist_lp_artist_ids",
        "lp_albums": "hist_lp_album_ids",
        "like_items": "hist_like_item_ids",
        "like_artists": "hist_like_artist_ids",
        "like_albums": "hist_like_album_ids",
        "skip_items": "hist_skip_item_ids",
        "skip_artists": "hist_skip_artist_ids",
        "skip_albums": "hist_skip_album_ids",
    }

    user_counters_np = None
    if enable_counters:
        user_counters_np = eval_dataset.get_user_counters(uid)

    for start in range(0, num_candidates, batch_size):
        end = min(start + batch_size, num_candidates)
        batch_items = candidate_item_ids[start:end]
        B = len(batch_items)

        batch = {
            "uid": torch.full((B,), uid, dtype=torch.long, device=device),
            "item_id": torch.from_numpy(batch_items).long().to(device),
            "artist_id": torch.from_numpy(
                eval_dataset.item_to_artist[batch_items]
            ).long().to(device),
            "album_id": torch.from_numpy(
                eval_dataset.item_to_album[batch_items]
            ).long().to(device),
            "audio_embed": torch.from_numpy(
                eval_dataset.audio_embeddings[batch_items]
            ).float().to(device),
        }

        for hk in hist_keys:
            batch[batch_key_map[hk]] = torch.from_numpy(
                np.tile(history[hk], (B, 1))
            ).long().to(device)

        if enable_counters:
            batch["user_counters"] = torch.from_numpy(
                np.tile(user_counters_np, (B, 1))
            ).float().to(device)
            batch["item_counters"] = torch.from_numpy(
                eval_dataset.get_item_counters_batch(batch_items)
            ).float().to(device)
            batch["cross_counters"] = torch.from_numpy(
                eval_dataset.get_cross_counters_batch(uid, batch_items)
            ).float().to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(batch)

        task_scores = preds[task].float().cpu().numpy()
        for i, item_id in enumerate(batch_items):
            scores.append((int(item_id), float(task_scores[i])))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
