"""Prepare VizWiz-VQA from Hugging Face into the local Arrow dataset format."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from vlm_uncertainty.datasets.arrow import save_vl_arrow


DEFAULT_DATASET_ID = "lmms-lab/VizWiz-VQA"
DEFAULT_PROMPT_TEMPLATE = "Return only the short sentence. Do not explain.\n\nQuestion: {question}\nAnswer:"


def majority_answer(answers: Iterable[str]) -> str | None:
    normalized = [answer.strip() for answer in answers if isinstance(answer, str) and answer.strip()]
    if not normalized:
        return None
    counts = Counter(normalized)
    return max(counts.items(), key=lambda item: (item[1], -normalized.index(item[0])))[0]


def _image_extension(image: Any) -> str:
    image_format = getattr(image, "format", None)
    if isinstance(image_format, str) and image_format:
        return f".{image_format.lower().replace('jpeg', 'jpg')}"
    return ".jpg"


def _save_image(image: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(image, "mode", None) not in {None, "RGB"}:
        image = image.convert("RGB")
    image.save(output_path)


def vizwiz_record(
    example: dict[str, Any],
    *,
    image_path: str,
    dataset_id: str = DEFAULT_DATASET_ID,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> dict[str, Any]:
    question = str(example.get("question", "")).strip()
    question_id = str(example.get("question_id", "")).strip()
    answers = example.get("answers") or []
    if not isinstance(answers, list):
        answers = list(answers)

    record: dict[str, Any] = {
        "id": question_id or Path(image_path).stem,
        "image": image_path,
        "prompt": prompt_template.format(question=question),
        "metadata": {
            "dataset": dataset_id,
            "question": question,
            "answers": answers,
            "category": example.get("category"),
        },
    }
    target = majority_answer(answers)
    if target is not None:
        record["target"] = target
    return record


def prepare_vizwiz_vqa(
    *,
    output_path: str | Path,
    image_dir: str | Path,
    split: str = "val",
    dataset_id: str = DEFAULT_DATASET_ID,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Preparing VizWiz-VQA requires `datasets`. Run `uv sync` first.") from exc

    dataset = load_dataset(dataset_id, split=split, trust_remote_code=trust_remote_code)
    image_root = Path(image_dir)
    output_parent = Path(output_path).parent
    records: list[dict[str, Any]] = []

    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break

        image = example["image"]
        question_id = str(example.get("question_id") or index)
        image_filename = f"{question_id}{_image_extension(image)}"
        absolute_image_path = image_root / split / image_filename
        _save_image(image, absolute_image_path)
        try:
            image_path = str(absolute_image_path.relative_to(output_parent))
        except ValueError:
            image_path = str(absolute_image_path)

        records.append(
            vizwiz_record(
                example,
                image_path=image_path,
                dataset_id=dataset_id,
                prompt_template=prompt_template,
            )
        )

    save_vl_arrow(output_path, records)
    return len(records)
