"""CLI for batch inference over prepared datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from vlm_uncertainty.datasets import load_vl_arrow
from vlm_uncertainty.inference.batch import run_batch_inference
from vlm_uncertainty.inference.predict import GenerationConfig, QwenVLInference
from vlm_uncertainty.models.qwen25_vl import DEFAULT_MODEL_ID
from vlm_uncertainty.utils.io import write_json


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _pick(cli_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL inference over a prepared dataset.")
    parser.add_argument("--config", default="configs/experiment/qwen25_vl_3b_instruct.yaml", help="Experiment YAML config path.")
    parser.add_argument("--input", default=None, help="Input Arrow dataset directory.")
    parser.add_argument("--output", default=None, help="Output JSON file path.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Number of examples per model.generate batch.")
    parser.add_argument("--n-generation", type=int, default=None, help="Number of generation calls per example.")
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--no-sample", action="store_false", dest="do_sample")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    model_config = config.get("model", {}) or {}
    generation_config = config.get("generation", {}) or {}
    data_config = config.get("data", {}) or {}

    input_dataset = _pick(args.input, data_config, "input_dataset")
    output_json = _pick(args.output, data_config, "output_json")
    if input_dataset is None:
        raise ValueError("Input dataset is required. Pass --input or set data.input_dataset in config.")
    if output_json is None:
        raise ValueError("Output JSON path is required. Pass --output or set data.output_json in config.")

    dataset = load_vl_arrow(Path(input_dataset))
    engine = QwenVLInference(
        model_id=_pick(args.model_id, model_config, "id", DEFAULT_MODEL_ID),
        device_map=_pick(args.device_map, model_config, "device_map", "auto"),
        torch_dtype=_pick(args.torch_dtype, model_config, "torch_dtype", "auto"),
        attn_implementation=_pick(args.attn_implementation, model_config, "attn_implementation"),
        min_pixels=_pick(args.min_pixels, model_config, "min_pixels"),
        max_pixels=_pick(args.max_pixels, model_config, "max_pixels"),
    )
    rows = run_batch_inference(
        engine,
        dataset,
        generation_config=GenerationConfig(
            max_new_tokens=_pick(args.max_new_tokens, generation_config, "max_new_tokens", 128),
            do_sample=_pick(args.do_sample, generation_config, "do_sample", False),
            temperature=_pick(args.temperature, generation_config, "temperature"),
            top_p=_pick(args.top_p, generation_config, "top_p"),
        ),
        system_prompt=args.system_prompt,
        n_generation=_pick(args.n_generation, generation_config, "n_generation", 1),
        batch_size=_pick(args.batch_size, generation_config, "batch_size", 1),
    )
    write_json(output_json, rows)


if __name__ == "__main__":
    main()
