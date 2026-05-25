"""Qwen2.5-VL inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from vlm_uncertainty.datasets.preprocessing import build_messages, normalize_image_ref
from vlm_uncertainty.models.qwen25_vl import DEFAULT_MODEL_ID, load_qwen25_vl


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs


class QwenVLInference:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        bundle = load_qwen25_vl(
            model_id=model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = bundle.model
        self.processor = bundle.processor

    def generate_from_messages_batch(
        self, messages_batch: list[list[dict]], config: GenerationConfig | None = None
    ) -> list[str]:
        config = config or GenerationConfig()
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Qwen vision preprocessing requires qwen-vl-utils and torchvision. "
                "Run `uv sync` to install the project dependencies."
            ) from exc

        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._input_device())
        generated_ids = self.model.generate(**inputs, **config.to_kwargs())
        generated_ids_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [output.strip() for output in decoded]

    def generate_from_messages(self, messages: list[dict], config: GenerationConfig | None = None) -> str:
        return self.generate_from_messages_batch([messages], config=config)[0]

    def generate(
        self,
        *,
        images: list[str],
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        normalized_images = [normalize_image_ref(image) for image in images]
        messages = build_messages(prompt=prompt, images=normalized_images, system_prompt=system_prompt)
        return self.generate_from_messages(messages, config=config)

    def _input_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"
