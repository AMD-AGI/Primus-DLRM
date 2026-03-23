#!/usr/bin/env python3
"""Full-catalog evaluation with multiple modes: global, peruser, mostpop."""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import YambdaEvalDataset
from primus_dlrm.evaluation.metrics import (
    evaluate_ranking_peruser,
    ndcg_at_k,
    recall_at_k,
)
from primus_dlrm.models.dlrm import DLRMBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def precompute_item_features(
    model: DLRMBaseline,
    candidate_item_ids: np.ndarray,
    eval_dataset: YambdaEvalDataset,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Precompute item tower representations for all candidate items."""
    model.eval()
    all_features = []
    enable_counters = eval_dataset.config.enable_counters

    for start in range(0, len(candidate_item_ids), batch_size):
        end = min(start + batch_size, len(candidate_item_ids))
        batch_items = candidate_item_ids[start:end]

        batch = {
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

        if enable_counters:
            batch["item_counters"] = torch.from_numpy(
                eval_dataset.get_item_counters_batch(batch_items)
            ).float().to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            item_feat = model._item_tower(batch)
        all_features.append(item_feat)

        if start % (batch_size * 20) == 0:
            logger.info(f"  Item precompute: {start}/{len(candidate_item_ids)}")

    return torch.cat(all_features, dim=0)


HIST_KEYS = [
    "lp_items", "lp_artists", "lp_albums",
    "like_items", "like_artists", "like_albums",
    "skip_items", "skip_artists", "skip_albums",
]
BATCH_KEY_MAP = {
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


def score_user_fast(
    model: DLRMBaseline,
    user_history: dict,
    uid: int,
    item_features: torch.Tensor,
    candidate_item_ids: np.ndarray,
    eval_dataset: YambdaEvalDataset,
    device: torch.device,
    task: str,
    batch_size: int = 8192,
) -> list[tuple[int, float]]:
    """Score all candidates for one user using precomputed item features."""
    enable_counters = eval_dataset.config.enable_counters

    user_batch = {"uid": torch.tensor([uid], dtype=torch.long, device=device)}
    for hk in HIST_KEYS:
        user_batch[BATCH_KEY_MAP[hk]] = torch.from_numpy(
            user_history[hk]
        ).long().unsqueeze(0).to(device)

    if enable_counters:
        user_counters_np = eval_dataset.get_user_counters(uid)
        user_batch["user_counters"] = torch.from_numpy(
            user_counters_np
        ).float().unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        user_feat = model._user_tower(user_batch)

    scores = []
    num_items = len(candidate_item_ids)

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        B = end - start

        user_tiled = user_feat.expand(B, -1)
        item_chunk = item_features[start:end]

        item_feats_list = [item_chunk]
        if enable_counters and model.num_counter_windows > 0:
            batch_items = candidate_item_ids[start:end]
            cross_np = eval_dataset.get_cross_counters_batch(uid, batch_items)
            cross_proj = model.cross_proj(
                torch.from_numpy(cross_np).float().to(device)
            )
            item_feats_list.append(cross_proj)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            interaction_out = model.interaction(
                user_features=[user_tiled],
                item_features=item_feats_list,
            )
            preds = {t: h(interaction_out).squeeze(-1) for t, h in model.heads.items()}
            logits = preds[task]

        task_scores = logits.float().cpu().numpy()
        for i, item_id in enumerate(candidate_item_ids[start:end]):
            scores.append((int(item_id), float(task_scores[i])))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def load_model(config, checkpoint_path, eval_dataset, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ckpt["model_state_dict"]
    num_users = sd["uid_emb.weight"].shape[0]
    num_items = sd["item_emb.weight"].shape[0]
    num_artists = sd["artist_emb.weight"].shape[0]
    num_albums = sd["album_emb.weight"].shape[0]
    tasks = sorted({k.split(".")[1] for k in sd if k.startswith("heads.")})

    num_counter_windows = 0
    if config.data.enable_counters:
        num_counter_windows = len(config.data.counter_windows_days)

    logger.info(
        f"From checkpoint: users={num_users}, items={num_items}, "
        f"artists={num_artists}, albums={num_albums}, tasks={tasks}, "
        f"counter_windows={num_counter_windows}"
    )

    model = DLRMBaseline(
        config=config.model,
        num_users=num_users,
        num_items=num_items,
        num_artists=num_artists,
        num_albums=num_albums,
        audio_input_dim=eval_dataset.audio_dim,
        device=device,
        tasks=tasks,
        num_counter_windows=num_counter_windows,
    )
    model.load_state_dict(sd)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def eval_mostpop(eval_dataset, task, ks, top_n=100):
    """MostPop baseline: rank by global listen_plus popularity, same list for all users."""
    t0 = time.time()
    store = eval_dataset.store
    lp_mask = store.flat_is_listen_plus
    item_ids = store.flat_item_ids[lp_mask]
    pop_counts = np.bincount(item_ids, minlength=len(eval_dataset.item_popularity))
    top_items = np.argsort(-pop_counts)[:top_n]
    ranked = [(int(iid), float(pop_counts[iid])) for iid in top_items]

    all_ndcg = {k: [] for k in ks}
    all_recall = {k: [] for k in ks}

    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        ground_truth = sample.get(f"ground_truth_{task}", set())
        if len(ground_truth) == 0:
            continue
        for k in ks:
            all_ndcg[k].append(ndcg_at_k(ranked, ground_truth, k))
            all_recall[k].append(recall_at_k(ranked, ground_truth, k))

    elapsed = time.time() - t0
    num_users = len(all_ndcg[ks[0]])
    logger.info(f"\n=== Mode: mostpop | Task: {task} | Users: {num_users} | Time: {elapsed:.1f}s ===")
    for k in ks:
        ndcg = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        recall = np.mean(all_recall[k]) if all_recall[k] else 0.0
        logger.info(f"  NDCG@{k}={ndcg:.4f} | Recall@{k}={recall:.4f}")


def eval_peruser(model, eval_dataset, device, task, ks, top_n=100, batch_size=1024):
    """Per-user eval: candidate pool = user's unique training items, top-N by score."""
    t0 = time.time()
    results = evaluate_ranking_peruser(
        model, eval_dataset, device,
        ks=ks, task=task, top_n=top_n, batch_size=batch_size,
    )
    elapsed = time.time() - t0
    logger.info(f"\n=== Mode: peruser | Task: {task} | Users: {results['num_users']} | Time: {elapsed:.1f}s ===")
    for k in ks:
        logger.info(f"  NDCG@{k}={results[f'ndcg@{k}']:.4f} | Recall@{k}={results[f'recall@{k}']:.4f}")


def eval_global(model, eval_dataset, device, task, ks, eval_top_k=0, score_batch_size=8192):
    """Global eval: precomputed item tower on a global candidate set."""
    pop = eval_dataset.item_popularity
    if eval_top_k > 0:
        candidate_items = np.argsort(-pop)[:eval_top_k]
        logger.info(f"Global eval: top-{eval_top_k} items")
    else:
        candidate_items = np.where(pop > 0)[0]
        logger.info(f"Global eval: full catalog ({len(candidate_items)} items)")

    logger.info("Precomputing item tower representations...")
    t0 = time.time()
    item_features = precompute_item_features(
        model, candidate_items, eval_dataset, device, batch_size=4096,
    )
    logger.info(f"Item precompute done: {item_features.shape} in {time.time() - t0:.1f}s")

    all_ndcg = {k: [] for k in ks}
    all_recall = {k: [] for k in ks}
    t_eval = time.time()

    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        uid = sample["uid"]
        history = sample["history"]
        ground_truth = sample.get(f"ground_truth_{task}", set())

        if len(ground_truth) == 0:
            continue

        user_scores = score_user_fast(
            model, history, uid, item_features,
            candidate_items, eval_dataset, device, task,
            batch_size=score_batch_size,
        )

        for k in ks:
            all_ndcg[k].append(ndcg_at_k(user_scores, ground_truth, k))
            all_recall[k].append(recall_at_k(user_scores, ground_truth, k))

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_eval
            logger.info(f"  Evaluated {idx + 1}/{len(eval_dataset)} users ({elapsed:.1f}s)")

    eval_time = time.time() - t_eval
    num_users = len(all_ndcg[ks[0]])
    logger.info(f"\n=== Mode: global | Task: {task} | Users: {num_users} | Time: {eval_time:.1f}s ===")
    for k in ks:
        ndcg = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        recall = np.mean(all_recall[k]) if all_recall[k] else 0.0
        logger.info(f"  NDCG@{k}={ndcg:.4f} | Recall@{k}={recall:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Full-catalog evaluation")
    parser.add_argument("--config", default="configs/exp_baseline_v1.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="results/baseline_v1/checkpoints/epoch_0.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["global", "peruser", "mostpop"], default="global",
                        help="Eval mode: global (top-K popular), peruser (per-user train items), mostpop (popularity baseline)")
    parser.add_argument("--task", default="listen_plus", choices=["listen_plus", "like"],
                        help="Which ground truth task to evaluate")
    parser.add_argument("--eval-top-k", type=int, default=0,
                        help="For global mode: 0 = full catalog, >0 = top-K popular items")
    parser.add_argument("--top-n", type=int, default=100,
                        help="For peruser/mostpop: number of top items to keep for metrics")
    parser.add_argument("--score-batch-size", type=int, default=8192)
    args = parser.parse_args()

    config = Config.load(args.config)
    device = torch.device(args.device)
    processed_dir = Path(args.processed_dir)
    ks = [10, 50, 100]

    logger.info("Loading eval dataset...")
    eval_dataset = YambdaEvalDataset(config.data, processed_dir)

    if args.mode == "mostpop":
        eval_mostpop(eval_dataset, args.task, ks, top_n=args.top_n)
    else:
        logger.info("Loading model checkpoint...")
        model = load_model(config, args.checkpoint, eval_dataset, device)

        if args.mode == "peruser":
            eval_peruser(model, eval_dataset, device, args.task, ks,
                         top_n=args.top_n, batch_size=args.score_batch_size)
        else:
            eval_global(model, eval_dataset, device, args.task, ks,
                        eval_top_k=args.eval_top_k, score_batch_size=args.score_batch_size)


if __name__ == "__main__":
    main()
