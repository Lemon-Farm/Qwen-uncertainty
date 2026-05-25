"""CLI for single Qwen2.5-VL inference."""

from __future__ import annotations

import argparse
import json

from vlm_uncertainty.inference.predict import GenerationConfig, QwenVLInference
from vlm_uncertainty.models.qwen25_vl import DEFAULT_MODEL_ID
from vlm_uncertainty.utils.io import read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL inference with user-provided images and prompt.")
    parser.add_argument("--image", action="append", required=True, help="Image path or URL. Repeat for multi-image input.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text.")
    prompt_group.add_argument("--prompt-file", help="Path to a text file containing the prompt.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-generation", type=int, default=1, help="Number of generation calls to run.")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of plain text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else read_text(args.prompt_file)
    engine = QwenVLInference(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if args.n_generation < 1:
        raise ValueError("n_generation must be at least 1")

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    messages = engine.build_messages(images=args.image, prompt=prompt, system_prompt=args.system_prompt)
    details = [
        engine.generate_from_messages_batch_with_details([messages], config=config)[0].to_dict()
        for _ in range(args.n_generation)
    ]
    outputs = [result["text"] for result in details]
    if args.json:
        print(
            json.dumps(
                {"prediction": outputs[0], "predictions": outputs, "prediction_details": details},
                ensure_ascii=False,
            )
        )
    else:
        print("\n".join(outputs))


if __name__ == "__main__":
    main()
