#!/usr/bin/env python3
"""Replicate the Yambda paper's MostPop baseline evaluation exactly.

Uses the paper's hardcoded constants and split logic to validate our eval pipeline.
Reference: Table 6 in https://arxiv.org/pdf/2505.22238
Expected results for Yambda-50M Listen+:
  NDCG@10=0.0186, NDCG@100=0.0249, Recall@10=0.0064, Recall@100=0.0321
"""
import argparse
import logging
import sys
import time

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Paper's hardcoded constants (from benchmarks/yambda/constants.py)
LAST_TIMESTAMP = 26_000_000
DAY_SECONDS = 24 * 60 * 60
GAP_SIZE = 30 * 60  # 30 minutes
TEST_SIZE = 1 * DAY_SECONDS
TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE  # 25913600
TRACK_LISTEN_THRESHOLD = 50


def ndcg_at_k(ranked_items: list[int], ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, item_id in enumerate(ranked_items[:k]):
        if item_id in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    ideal_len = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items: list[int], ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    top_k = set(ranked_items[:k])
    return len(top_k & ground_truth) / len(ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="data/raw/50m/multi_event.parquet")
    parser.add_argument("--task", default="listen_plus", choices=["listen_plus", "like"])
    parser.add_argument("--num-ranked", type=int, default=100)
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_file}...")
    t0 = time.time()
    df = pl.read_parquet(args.data_file)
    logger.info(f"Loaded {len(df):,} events in {time.time()-t0:.1f}s")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Timestamp range: [{df['timestamp'].min()}, {df['timestamp'].max()}]")

    # Filter to the relevant event type
    if args.task == "listen_plus":
        events = df.filter(
            (pl.col("event_type") == "listen") &
            (pl.col("played_ratio_pct") >= TRACK_LISTEN_THRESHOLD)
        )
        logger.info(f"Listen+ events: {len(events):,}")
    elif args.task == "like":
        events = df.filter(pl.col("event_type") == "like")
        logger.info(f"Like events: {len(events):,}")

    # Apply the paper's exact temporal split
    train_timestamp = TEST_TIMESTAMP - GAP_SIZE  # train: [0, TEST_TIMESTAMP - GAP_SIZE)
    logger.info(f"Split: train < {train_timestamp}, test >= {TEST_TIMESTAMP}")

    train = events.filter(pl.col("timestamp") < train_timestamp)
    test = events.filter(pl.col("timestamp") >= TEST_TIMESTAMP)
    logger.info(f"Train events: {len(train):,}, Test events: {len(test):,}")

    # Only keep test users that appear in training
    train_uids = set(train["uid"].unique().to_list())
    test = test.filter(pl.col("uid").is_in(list(train_uids)))
    logger.info(f"Test events (train users only): {len(test):,}")

    test_uids = test["uid"].unique().to_list()
    logger.info(f"Test users: {len(test_uids)}")

    # Build ground truth per user
    ground_truth = {}
    for row in test.iter_rows(named=True):
        uid = row["uid"]
        if uid not in ground_truth:
            ground_truth[uid] = set()
        ground_truth[uid].add(row["item_id"])

    # Only keep users with at least 1 ground truth item
    test_uids_with_gt = [uid for uid in test_uids if len(ground_truth.get(uid, set())) > 0]
    logger.info(f"Test users with ground truth: {len(test_uids_with_gt)}")

    # MostPop: count items in training, take top-K
    item_counts = train.group_by("item_id").agg(pl.len().alias("count")).sort("count", descending=True)
    top_items = item_counts["item_id"].to_list()[:args.num_ranked]
    logger.info(f"Top-{args.num_ranked} items by popularity (counts: {item_counts['count'][0]} .. {item_counts['count'][min(args.num_ranked-1, len(item_counts)-1)]})")

    # Evaluate: same ranked list for every user
    ks = [10, 50, 100]
    all_ndcg = {k: [] for k in ks}
    all_recall = {k: [] for k in ks}

    for uid in test_uids_with_gt:
        gt = ground_truth[uid]
        for k in ks:
            all_ndcg[k].append(ndcg_at_k(top_items, gt, k))
            all_recall[k].append(recall_at_k(top_items, gt, k))

    logger.info(f"\n{'='*60}")
    logger.info(f"MostPop Results — {args.task} (Yambda-50M)")
    logger.info(f"{'='*60}")
    logger.info(f"Users evaluated: {len(test_uids_with_gt)}")
    for k in ks:
        ndcg = np.mean(all_ndcg[k])
        rec = np.mean(all_recall[k])
        logger.info(f"  NDCG@{k:>3d} = {ndcg:.4f}  |  Recall@{k:>3d} = {rec:.4f}")
    logger.info(f"{'='*60}")

    # Paper reference (Table 6, Yambda-50M Listen+)
    if args.task == "listen_plus":
        logger.info("\nPaper reference (Table 6, Listen+):")
        logger.info("  NDCG@10  = 0.0186  |  Recall@10  = 0.0064")
        logger.info("  NDCG@100 = 0.0249  |  Recall@100 = 0.0321")
    elif args.task == "like":
        logger.info("\nPaper reference (Table 7, Like):")
        logger.info("  NDCG@10  = 0.0046  |  Recall@10  = 0.0083")
        logger.info("  NDCG@100 = 0.0097  |  Recall@100 = 0.0222")


if __name__ == "__main__":
    main()
