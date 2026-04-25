#!/usr/bin/env python3
"""Preprocess Yambda data: temporal split, sessions, metadata.

The default output directory is derived from ``--size`` so the variants
never collide:

  --size 50m   ->  data/processed/      (legacy default kept for back-compat)
  --size 500m  ->  data/processed_500m/
  --size 5b    ->  data/processed_5b/

Override with an explicit ``--out-dir`` if you want something else.
"""
import argparse
import logging

from primus_dlrm.data.preprocessing import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_OUT_DIR_BY_SIZE = {
    "50m": "data/processed",
    "500m": "data/processed_500m",
    "5b": "data/processed_5b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Yambda dataset")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: derived from --size, e.g. "
             "data/processed for 50m, data/processed_5b for 5b)",
    )
    parser.add_argument("--size", default="50m", choices=["50m", "500m", "5b"])
    parser.add_argument("--session-gap", type=int, default=1800,
                        help="Session gap in seconds")
    parser.add_argument("--train-days", type=int, default=300)
    parser.add_argument("--gap-minutes", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument(
        "--no-chunked", action="store_true",
        help="Disable chunked load (use eager read_parquet). Chunked is "
             "default since it is required for the 5b dataset and harmless "
             "for smaller sizes.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or _OUT_DIR_BY_SIZE[args.size]
    preprocess(
        raw_dir=args.raw_dir,
        out_dir=out_dir,
        dataset_size=args.size,
        session_gap_seconds=args.session_gap,
        train_days=args.train_days,
        gap_minutes=args.gap_minutes,
        test_days=args.test_days,
        chunked=not args.no_chunked,
    )


if __name__ == "__main__":
    main()
