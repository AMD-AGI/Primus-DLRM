"""Download Yambda dataset from HuggingFace."""
from __future__ import annotations

import logging
from pathlib import Path

from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)

REPO_ID = "yandex/yambda"

_INTERACTION_FILES = [
    "multi_event",
    "likes",
    "listens",
    "dislikes",
    "unlikes",
    "undislikes",
]

_METADATA_FILES = [
    "embeddings",
    "artist_item_mapping",
    "album_item_mapping",
]


def download_yambda(
    data_dir: str | Path = "data/raw",
    dataset_size: str = "50m",
    files: list[str] | None = None,
) -> Path:
    """Download Yambda dataset files to local directory.

    Args:
        data_dir: Root directory for raw data.
        dataset_size: One of "50m", "500m", "5b".
        files: Specific files to download. If None, downloads
               multi_event + metadata files (embeddings, mappings).

    Returns:
        Path to the data directory.
    """
    assert dataset_size in {"50m", "500m", "5b"}
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if files is None:
        files = ["multi_event"] + _METADATA_FILES

    for file_name in files:
        if file_name in _METADATA_FILES:
            hf_data_dir = ""
            out_path = data_dir / f"{file_name}.parquet"
        else:
            hf_data_dir = f"flat/{dataset_size}"
            out_path = data_dir / dataset_size / f"{file_name}.parquet"

        if out_path.exists():
            logger.info(f"Already exists: {out_path}")
            continue

        logger.info(f"Downloading {file_name} from {REPO_ID} ({hf_data_dir or 'root'})...")
        ds = load_dataset(
            REPO_ID,
            data_dir=hf_data_dir if hf_data_dir else None,
            data_files=f"{file_name}.parquet",
        )
        assert isinstance(ds, DatasetDict)
        table = ds["train"]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(str(out_path))
        logger.info(f"Saved {len(table)} rows to {out_path}")

    return data_dir
