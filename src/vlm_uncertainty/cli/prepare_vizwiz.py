"""CLI for preparing VizWiz-VQA Arrow datasets."""

from __future__ import annotations

import argparse

from vlm_uncertainty.datasets.vizwiz import (
    DEFAULT_DATASET_ID,
    DEFAULT_PROMPT_TEMPLATE,
    prepare_vizwiz_vqa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare lmms-lab/VizWiz-VQA into the local Arrow dataset format.")
    parser.add_argument("--split", default="val", help="Hugging Face split to prepare, e.g. val or test.")
    parser.add_argument("--output", default="data/vizwiz_val", help="Output Arrow dataset directory.")
    parser.add_argument("--image-dir", default="data/vizwiz_images", help="Directory where images will be saved.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset id.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template. Use {question} where the VizWiz question should appear.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of examples for smoke tests.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = prepare_vizwiz_vqa(
        output_path=args.output,
        image_dir=args.image_dir,
        split=args.split,
        dataset_id=args.dataset_id,
        prompt_template=args.prompt_template,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote {count} examples to Arrow dataset {args.output}")


if __name__ == "__main__":
    main()
