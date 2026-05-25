"""Qwen2.5-VL model and processor loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass(frozen=True)
class QwenVLBundle:
    model: Any
    processor: Any


def load_qwen25_vl(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device_map: str | None = "auto",
    torch_dtype: str = "auto",
    attn_implementation: str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> QwenVLBundle:
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Install inference dependencies first, for example: `uv sync`. "
            "Qwen2.5-VL requires a recent transformers build."
        ) from exc

    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device_map:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    processor_kwargs = {k: v for k, v in {"min_pixels": min_pixels, "max_pixels": max_pixels}.items() if v is not None}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return QwenVLBundle(model=model, processor=processor)
