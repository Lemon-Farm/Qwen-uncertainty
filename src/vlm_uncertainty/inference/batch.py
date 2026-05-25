"""Batch inference helpers."""

from __future__ import annotations

from typing import Iterable

from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm_uncertainty.datasets.base import VLExample
from vlm_uncertainty.inference.predict import GenerationConfig, QwenVLInference


def _collate_examples(examples: list[VLExample]) -> list[VLExample]:
    return examples


def run_batch_inference(
    engine: QwenVLInference,
    examples: Iterable[VLExample],
    *,
    generation_config: GenerationConfig | None = None,
    system_prompt: str | None = None,
    n_generation: int = 1,
    batch_size: int = 1,
) -> list[dict]:
    if n_generation < 1:
        raise ValueError("n_generation must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    example_list = list(examples)
    loader = DataLoader(
        example_list,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_examples,
    )
    rows: list[dict] = []

    for batch in tqdm(loader, desc="inference"):
        messages_batch = [example.to_messages(system_prompt=system_prompt) for example in batch]
        prediction_details_by_example: list[list[dict]] = [[] for _ in batch]

        for _ in range(n_generation):
            batch_results = engine.generate_from_messages_batch_with_details(
                messages_batch,
                config=generation_config,
            )
            for prediction_details, result in zip(prediction_details_by_example, batch_results):
                prediction_details.append(result.to_dict())

        for example, prediction_details in zip(batch, prediction_details_by_example):
            predictions = [result["text"] for result in prediction_details]
            rows.append(
                {
                    "id": example.id,
                    "prompt": example.prompt,
                    "images": example.images,
                    "prediction": predictions[0],
                    "predictions": predictions,
                    "prediction_details": prediction_details,
                    "target": example.target,
                    "metadata": example.metadata,
                }
            )
    return rows
