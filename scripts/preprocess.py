#!/usr/bin/env python3
"""Preprocess Yambda data: temporal split, sessions, metadata."""
import argparse
import logging

from primus_dlrm.data.preprocessing import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Yambda dataset")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    parser.add_argument("--size", default="50m", choices=["50m", "500m", "5b"])
    parser.add_argument("--session-gap", type=int, default=1800, help="Session gap in seconds")
    parser.add_argument("--train-days", type=int, default=300)
    parser.add_argument("--gap-minutes", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=1)
    args = parser.parse_args()

    preprocess(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        dataset_size=args.size,
        session_gap_seconds=args.session_gap,
        train_days=args.train_days,
        gap_minutes=args.gap_minutes,
        test_days=args.test_days,
    )


if __name__ == "__main__":
    main()
