"""Lightweight metrics for generated text."""

from __future__ import annotations


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def exact_match(prediction: str, target: str) -> float:
    return float(normalize_text(prediction) == normalize_text(target))
