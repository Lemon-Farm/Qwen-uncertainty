"""Prepare TextVQA from Hugging Face into the local Arrow dataset format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_uncertainty.datasets.arrow import save_vl_arrow


DEFAULT_DATASET_ID = "lmms-lab/textvqa"
DEFAULT_PROMPT_TEMPLATE = "Return only the short sentence. Do not explain.\n\nQuestion: {question}\nAnswer:"



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


def textvqa_record(
    example: dict[str, Any],
    *,
    image_path: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> dict[str, str]:
    question_id = str(example.get("question_id", "")).strip() or Path(image_path).stem
    question = str(example.get("question", "")).strip()
    return {
        "question_id": question_id,
        "question": prompt_template.format(question=question),
        "image": image_path,
    }


def prepare_textvqa(
    *,
    output_path: str | Path,
    image_dir: str | Path,
    split: str = "validation",
    dataset_id: str = DEFAULT_DATASET_ID,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Preparing TextVQA requires `datasets`. Run `uv sync` first.") from exc

    dataset_split = f"{split}[:{limit}]" if limit is not None else split
    dataset = load_dataset(dataset_id, split=dataset_split, trust_remote_code=trust_remote_code)
    dataset = dataset.select_columns(["question_id", "question", "image"])

    image_root = Path(image_dir)
    output_parent = Path(output_path).parent
    records: list[dict[str, str]] = []

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
            textvqa_record(
                example,
                image_path=image_path,
                prompt_template=prompt_template,
            )
        )

    save_vl_arrow(output_path, records)
    return len(records)
