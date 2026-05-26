"""Semantic entropy calculation from generation logprobs and semantic clusters."""

from __future__ import annotations

import math
from typing import Any


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _prediction_logprobs(row: dict[str, Any]) -> list[float]:
    details = row.get("prediction_details")
    if isinstance(details, list) and details:
        logprobs = []
        for item in details:
            if isinstance(item, dict) and item.get("logprob") is not None:
                logprobs.append(float(item["logprob"]))
        if logprobs:
            return logprobs

    predictions = row.get("predictions") or []
    if not predictions:
        return []
    uniform_logprob = -math.log(len(predictions))
    return [uniform_logprob for _ in predictions]


def semantic_entropy_from_clusters(logprobs: list[float], clusters: list[list[int]]) -> float:
    if not logprobs or not clusters:
        return 0.0

    normalizer = logsumexp(logprobs)
    entropy = 0.0
    for cluster in clusters:
        cluster_logprobs = [logprobs[index] for index in cluster if 0 <= index < len(logprobs)]
        if not cluster_logprobs:
            continue
        cluster_prob = math.exp(logsumexp(cluster_logprobs) - normalizer)
        if cluster_prob > 0.0:
            entropy -= cluster_prob * math.log(cluster_prob)
    return entropy


def calculate_semantic_entropy_rows(
    *,
    predictions: list[dict[str, Any]],
    clustered_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions_by_id = {str(row.get("id")): row for row in predictions}
    output_rows: list[dict[str, Any]] = []

    for clustered_row in clustered_predictions:
        row_id = str(clustered_row.get("id"))
        prediction_row = predictions_by_id.get(row_id, clustered_row)
        logprobs = _prediction_logprobs(prediction_row)
        clusters = clustered_row.get("semantic_clusters") or []
        entropy = semantic_entropy_from_clusters(logprobs, clusters)

        output_rows.append(
            {
                "id": clustered_row.get("id"),
                "prompt": clustered_row.get("prompt"),
                "images": clustered_row.get("images"),
                "predictions": clustered_row.get("predictions", prediction_row.get("predictions", [])),
                "sementic_entropy": entropy,
            }
        )
    return output_rows
