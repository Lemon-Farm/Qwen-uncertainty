"""CLI for calculating semantic entropy from clustered predictions."""

from __future__ import annotations

import argparse

from vlm_uncertainty.uncertainty.semantic_entropy import calculate_semantic_entropy_rows
from vlm_uncertainty.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate semantic entropy from prediction and cluster JSON files.")
    parser.add_argument("--predictions", default="outputs/predictions.json", help="Predictions JSON with logprobs.")
    parser.add_argument(
        "--clusters",
        default="outputs/predictions_semantic_clusters.json",
        help="Predictions JSON with semantic_clusters.",
    )
    parser.add_argument("--output", default="outputs/predictions_semantic_entropy.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_json(args.predictions)
    clustered_predictions = read_json(args.clusters)
    if not isinstance(predictions, list):
        raise ValueError(f"Predictions JSON must contain a list: {args.predictions}")
    if not isinstance(clustered_predictions, list):
        raise ValueError(f"Clusters JSON must contain a list: {args.clusters}")

    rows = calculate_semantic_entropy_rows(
        predictions=predictions,
        clustered_predictions=clustered_predictions,
    )
    write_json(args.output, rows)
    print(f"Wrote semantic entropy rows to {args.output}")


if __name__ == "__main__":
    main()
