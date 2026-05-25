"""Apply image perturbations to prepared Arrow datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image, ImageFilter
from tqdm import tqdm

from vlm_uncertainty.datasets.arrow import save_vl_arrow
from vlm_uncertainty.datasets.preprocessing import is_url


def _load_arrow_records(path: str | Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Perturbing Arrow datasets requires `datasets`. Run `uv sync` first.") from exc

    dataset = load_from_disk(str(path))
    return [dict(record) for record in dataset]


def _resolve_image_path(image_ref: str, *, base_dir: Path) -> Path:
    if is_url(image_ref):
        raise ValueError(f"Remote image URLs are not supported for perturbation: {image_ref}")
    if image_ref.startswith("file://"):
        parsed = urlparse(image_ref)
        return Path(unquote(parsed.path)).expanduser().resolve()
    path = Path(image_ref)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def _relative_source_path(source_path: Path, *, base_dir: Path) -> Path:
    try:
        return source_path.relative_to(base_dir.resolve())
    except ValueError:
        return Path(source_path.name)


def _stored_image_ref(output_path: Path, *, output_base_dir: Path) -> str:
    try:
        return str(output_path.relative_to(output_base_dir))
    except ValueError:
        return str(output_path)


def _apply_gaussian_blur(source_path: Path, output_path: Path, *, radius: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.filter(ImageFilter.GaussianBlur(radius=radius)).save(output_path)


def _perturb_image_ref(
    image_ref: str,
    *,
    input_base_dir: Path,
    output_base_dir: Path,
    output_image_dir: Path,
    perturbation_name: str,
    radius: float,
) -> str:
    source_path = _resolve_image_path(image_ref, base_dir=input_base_dir)
    relative_path = _relative_source_path(source_path, base_dir=input_base_dir)
    output_path = output_image_dir / perturbation_name / relative_path
    _apply_gaussian_blur(source_path, output_path, radius=radius)
    return _stored_image_ref(output_path, output_base_dir=output_base_dir)


def apply_image_perturbation_dataset(
    *,
    input_dataset: str | Path,
    output_dataset: str | Path,
    output_image_dir: str | Path,
    perturbation: dict[str, Any],
) -> int:
    name = perturbation.get("name")
    if name != "gaussian_blur":
        raise ValueError(f"Unsupported perturbation: {name}")
    radius = float(perturbation.get("radius", 5))

    input_dataset_path = Path(input_dataset)
    output_dataset_path = Path(output_dataset)
    input_base_dir = input_dataset_path.parent
    output_base_dir = output_dataset_path.parent
    output_image_root = Path(output_image_dir)

    records = _load_arrow_records(input_dataset_path)
    perturbed_records: list[dict[str, Any]] = []

    for record in tqdm(records, desc="perturb images"):
        updated = dict(record)
        if "image" in updated:
            updated["image"] = _perturb_image_ref(
                str(updated["image"]),
                input_base_dir=input_base_dir,
                output_base_dir=output_base_dir,
                output_image_dir=output_image_root,
                perturbation_name=name,
                radius=radius,
            )
        elif "images" in updated:
            updated["images"] = [
                _perturb_image_ref(
                    str(image),
                    input_base_dir=input_base_dir,
                    output_base_dir=output_base_dir,
                    output_image_dir=output_image_root,
                    perturbation_name=name,
                    radius=radius,
                )
                for image in updated["images"]
            ]
        else:
            raise ValueError("Each record must contain `image` or `images`")
        perturbed_records.append(updated)

    save_vl_arrow(output_dataset_path, perturbed_records)
    return len(perturbed_records)
