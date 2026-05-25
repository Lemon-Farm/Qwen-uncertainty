# VLM Uncertainty

Qwen2.5-VL 기반 VQA/Document VQA 응답의 uncertainty를 측정하기 위한 실험 레포.
현재 목표는 이미지 perturbation 전후의 VLM 답변 변화를 보고, Semantic Uncertainty 방식으로 의미 단위 불확실성을 계산하는 것이다.

## Experiment

기본 흐름:

1. Hugging Face VQA dataset을 Arrow dataset으로 준비한다.
2. Qwen2.5-VL-3B-Instruct로 한 질문당 여러 답변을 sampling한다.
3. 각 답변의 token log probability를 저장한다.
4. NLI 모델로 답변들을 semantic cluster로 묶는다.
5. clean dataset과 perturbed dataset의 uncertainty를 비교한다.

## Current Setup

- VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
- Inference backend: `transformers`, `qwen-vl-utils`, `torch`
- Dataset format: Hugging Face Arrow dataset
- Output format: JSON
- NLI model: `cross-encoder/nli-deberta-v3-large`
- Implemented perturbation: Gaussian blur, radius `5`

## Implemented

Dataset prepare:

```bash
uv run prepare-vizwiz-vqa
uv run prepare-docvqa
uv run prepare-textvqa
```

Batch VLM inference:

```bash
uv run vlm-batch-infer --config configs/experiment/qwen25_vl_3b_instruct.yaml
```

Image perturbation:

```bash
uv run perturb-dataset --config configs/perturbation/gaussian_blur.yaml
```

NLI semantic clustering:

```bash
uv run vlm-semantic-cluster --config configs/uncertainty/semantic_clustering.yaml
```

Current inference output includes:

- generated answers
- per-answer log probability
- normalized log probability
- semantic clusters

## Configs

- `configs/experiment/qwen25_vl_3b_instruct.yaml`: model, generation, input/output paths
- `configs/perturbation/gaussian_blur.yaml`: perturbation target dataset and blur setting
- `configs/uncertainty/semantic_clustering.yaml`: NLI model and clustering settings

## Todo

- Implement semantic entropy calculation from semantic clusters and answer logprobs.
- Add clean vs perturbed uncertainty comparison script.
- Add aggregation metrics across datasets.
- Add reproducible seed handling for generation.
- Add evaluation helpers for datasets with ground-truth answers.
- Add more perturbations beyond Gaussian blur.
- Add lightweight tests for CLI configs and output schemas.
