# vlm-uncertainty

Project skeleton for running Qwen2.5-VL inference and experiment data pipelines.

## Single-image inference

Provide your own image and prompt:

```bash
uv run vlm-infer --image /path/to/image.jpg --prompt "Your prompt here"
```

Or keep the prompt in a local text file:

```bash
uv run vlm-infer --image /path/to/image.jpg --prompt-file /path/to/prompt.txt
```

## JSONL dataset format

Each line should contain one example. Paths are resolved relative to the JSONL file unless they are absolute paths, URLs, or `file://` URIs.

```json
{"id": "sample-001", "image": "images/sample.jpg", "prompt": "Your prompt here", "target": "optional reference answer"}
```

Batch inference:

```bash
uv run vlm-batch-infer --input data/my_dataset.jsonl --output outputs/predictions.jsonl
```
