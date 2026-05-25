"""Collators for supervised VLM fine-tuning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from vlm_uncertainty.datasets.base import VLExample


@dataclass
class QwenVLSupervisedCollator:
    processor: Any
    system_prompt: str | None = None

    def __call__(self, examples: list[VLExample]) -> dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Qwen vision preprocessing requires qwen-vl-utils and torchvision. "
                "Run `uv sync` to install the project dependencies."
            ) from exc

        messages_batch = [example.to_messages(system_prompt=self.system_prompt) for example in examples]
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for messages in messages_batch:
            images, videos = process_vision_info(messages)
            image_inputs.extend(images or [])
            video_inputs.extend(videos or [])
        batch = self.processor(
            text=texts,
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
