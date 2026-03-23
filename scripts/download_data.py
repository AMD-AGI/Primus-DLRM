#!/usr/bin/env python3
"""Download Yambda dataset from HuggingFace."""
import argparse
import logging

from primus_dlrm.data.download import download_yambda

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Download Yambda dataset")
    parser.add_argument("--data-dir", default="data/raw", help="Output directory")
    parser.add_argument("--size", default="50m", choices=["50m", "500m", "5b"])
    args = parser.parse_args()

    download_yambda(data_dir=args.data_dir, dataset_size=args.size)


if __name__ == "__main__":
    main()
