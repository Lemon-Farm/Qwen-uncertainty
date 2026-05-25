"""CLI for preparing DocVQA Arrow datasets."""

from __future__ import annotations

import argparse

from vlm_uncertainty.datasets.docvqa import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_PROMPT_TEMPLATE,
    prepare_docvqa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare lmms-lab/DocVQA into the local Arrow dataset format.")
    parser.add_argument("--split", default="validation", help="Hugging Face split to prepare, e.g. validation or test.")
    parser.add_argument("--output", default="data/docvqa_validation", help="Output Arrow dataset directory.")
    parser.add_argument("--image-dir", default="data/docvqa_images", help="Directory where images will be saved.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset id.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Hugging Face dataset config name.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template. Use {question} where the DocVQA question should appear.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of examples for smoke tests.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name if args.dataset_name else None
    count = prepare_docvqa(
        output_path=args.output,
        image_dir=args.image_dir,
        split=args.split,
        dataset_id=args.dataset_id,
        dataset_name=dataset_name,
        prompt_template=args.prompt_template,
        limit=args.limit,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote {count} examples to Arrow dataset {args.output}")


if __name__ == "__main__":
    main()
