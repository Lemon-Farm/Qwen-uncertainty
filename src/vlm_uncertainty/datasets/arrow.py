"""Arrow-backed dataset support for image-prompt experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from vlm_uncertainty.datasets.base import VLDataset, VLExample


def save_vl_arrow(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Arrow dataset support requires `datasets`. Run `uv sync` first.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(list(records)).save_to_disk(str(output_path))


def load_vl_arrow(path: str | Path) -> VLDataset:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Arrow dataset support requires `datasets`. Run `uv sync` first.") from exc

    dataset_path = Path(path)
    dataset = load_from_disk(str(dataset_path))
    return VLDataset([VLExample.from_record(record, base_dir=dataset_path.parent) for record in dataset])


def is_arrow_dataset(path: str | Path) -> bool:
    dataset_path = Path(path)
    return dataset_path.is_dir() and (dataset_path / "dataset_info.json").exists()
