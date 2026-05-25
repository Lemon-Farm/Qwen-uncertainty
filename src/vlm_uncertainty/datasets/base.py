"""Common image-prompt dataset types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .preprocessing import build_messages, normalize_image_ref


@dataclass(frozen=True)
class VLExample:
    id: str
    images: list[str]
    prompt: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any], base_dir: str | Path | None = None) -> "VLExample":
        raw_images = record.get("images", record.get("image"))
        if raw_images is None:
            raise ValueError("Each record must contain `image` or `images`")
        if isinstance(raw_images, str):
            image_values = [raw_images]
        elif isinstance(raw_images, Sequence):
            image_values = list(raw_images)
        else:
            raise ValueError("`image` must be a string or `images` must be a list of strings")
        if not all(isinstance(image, str) for image in image_values):
            raise ValueError("All image values must be strings")

        prompt = record.get("prompt", record.get("question"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each record must contain a non-empty string `prompt` or `question`")

        raw_metadata = record.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.update(
            {
                k: v
                for k, v in record.items()
                if k not in {"id", "questionId", "question_id", "image", "images", "prompt", "question", "target", "metadata"}
            }
        )

        example_id = str(record.get("id", record.get("questionId", record.get("question_id", "")))) or "unknown"
        return cls(
            id=example_id,
            images=[normalize_image_ref(image, base_dir=base_dir) for image in image_values],
            prompt=prompt,
            target=record.get("target"),
            metadata=metadata,
        )

    def to_messages(self, system_prompt: str | None = None) -> list[dict]:
        return build_messages(prompt=self.prompt, images=self.images, system_prompt=system_prompt)


class VLDataset:
    def __init__(self, examples: Sequence[VLExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> VLExample:
        return self.examples[index]

    def __iter__(self) -> Iterator[VLExample]:
        return iter(self.examples)
