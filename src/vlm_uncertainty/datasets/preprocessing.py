"""Preprocessing helpers for Qwen-style vision-language messages."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def normalize_image_ref(image: str, base_dir: str | Path | None = None) -> str:
    if image.startswith("file://") or is_url(image):
        return image
    path = Path(image)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.expanduser().resolve().as_uri()


def build_user_message(prompt: str, images: list[str]) -> dict:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def build_messages(prompt: str, images: list[str], system_prompt: str | None = None) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append(build_user_message(prompt=prompt, images=images))
    return messages
