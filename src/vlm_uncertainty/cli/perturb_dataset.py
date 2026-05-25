"""CLI for applying image perturbations to prepared Arrow datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from vlm_uncertainty.perturbations import apply_image_perturbation_dataset


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply image perturbations to an Arrow dataset.")
    parser.add_argument("--config", required=False, default="configs/perturbation/gaussian_blur.yaml", help="Perturbation YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    data_config = config.get("data", {}) or {}
    perturbation_config = config.get("perturbation", {}) or {}

    input_dataset = data_config.get("input_dataset")
    output_dataset = data_config.get("output_dataset")
    output_image_dir = data_config.get("output_image_dir")
    if input_dataset is None:
        raise ValueError("data.input_dataset is required")
    if output_dataset is None:
        raise ValueError("data.output_dataset is required")
    if output_image_dir is None:
        raise ValueError("data.output_image_dir is required")

    count = apply_image_perturbation_dataset(
        input_dataset=input_dataset,
        output_dataset=output_dataset,
        output_image_dir=output_image_dir,
        perturbation=perturbation_config,
    )
    print(f"Wrote {count} perturbed examples to {output_dataset}")


if __name__ == "__main__":
    main()
