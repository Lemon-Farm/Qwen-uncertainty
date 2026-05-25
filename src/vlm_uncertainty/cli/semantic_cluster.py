"""CLI for NLI-based semantic clustering of generated answers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from vlm_uncertainty.uncertainty import cluster_prediction_rows
from vlm_uncertainty.uncertainty.semantic_clustering import SemanticClusteringConfig
from vlm_uncertainty.utils.io import read_json, write_json


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster generated answers by NLI semantic equivalence.")
    parser.add_argument("--config", required=False, default="configs/uncertainty/semantic_clustering.yaml", help="Semantic clustering YAML config path.")
    parser.add_argument("--input", default=None, help="Input predictions JSON path.")
    parser.add_argument("--output", default=None, help="Output clustered predictions JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    data_config = config.get("data", {}) or {}
    model_config = config.get("model", {}) or {}
    clustering_config = config.get("clustering", {}) or {}

    input_json = args.input or data_config.get("input_json")
    output_json = args.output or data_config.get("output_json")
    if input_json is None:
        raise ValueError("Input predictions JSON is required. Pass --input or set data.input_json.")
    if output_json is None:
        raise ValueError("Output JSON is required. Pass --output or set data.output_json.")

    rows = read_json(input_json)
    if not isinstance(rows, list):
        raise ValueError(f"Input predictions JSON must contain a list: {input_json}")

    semantic_config = SemanticClusteringConfig(
        model_id=model_config.get("id", "cross-encoder/nli-deberta-v3-large"),
        device=model_config.get("device"),
        batch_size=int(model_config.get("batch_size", 32)),
        max_length=int(model_config.get("max_length", 512)),
        entailment_threshold=clustering_config.get("entailment_threshold"),
        premise_template=clustering_config.get(
            "premise_template",
            "Question: {question}\nAnswer: {answer}",
        ),
        hypothesis_template=clustering_config.get(
            "hypothesis_template",
            "Question: {question}\nAnswer: {answer}",
        ),
        include_pair_scores=bool(clustering_config.get("include_pair_scores", False)),
    )
    clustered_rows = cluster_prediction_rows(rows, config=semantic_config)
    write_json(output_json, clustered_rows)
    print(f"Wrote clustered predictions to {output_json}")


if __name__ == "__main__":
    main()
