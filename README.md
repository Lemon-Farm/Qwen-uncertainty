# VLM Semantic Uncertainty

Qwen2.5-VL 기반 VQA/Document VQA 응답의 Semantic Uncertainty 실험 레포.
이미지 perturbation 전후로 VLM의 답변 분포가 얼마나 의미적으로 흔들리는지 측정하고, clean/perturbed 구분 성능을 ROC-AUC와 F1-score로 평가한다.

## Experiment Summary

실험 흐름은 다음과 같다.

1. VQA dataset을 Hugging Face Arrow dataset으로 준비한다.
2. 같은 질문에 대해 VLM 답변을 여러 번 sampling한다.
3. 각 답변의 token log probability를 저장한다.
4. NLI 모델로 답변들을 semantic cluster로 묶는다.
5. cluster 단위 확률로 Semantic Entropy를 계산한다.
6. clean dataset과 perturbed dataset의 Semantic Entropy를 비교한다.
7. ROC curve, AUC, best F1-score를 report로 저장한다.

## Setup

- VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
- NLI model: `cross-encoder/nli-deberta-v3-large`
- Dataset format: Hugging Face Arrow dataset
- Inference output: JSON
- Perturbation: Gaussian blur, radius `5`
- Package runner: `uv`

## Commands

Dataset prepare:

```bash
uv run prepare-vizwiz-vqa
uv run prepare-docvqa
uv run prepare-textvqa
```

VLM inference:

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

Semantic Entropy calculation:

```bash
uv run calculate-SE \
  --predictions outputs/predictions.json \
  --clusters outputs/predictions_semantic_clusters.json \
  --output outputs/predictions_semantic_entropy.json
```

Clean vs perturbed evaluation:

```bash
uv run evaluate-roc-f1 \
  --clean outputs/clean/predictions_semantic_entropy.json \
  --perturbed outputs/perturbed/predictions_semantic_entropy.json \
  --report-dir reports/docvqa
```

## Configs

- `configs/experiment/qwen25_vl_3b_instruct.yaml`: model, generation, input/output path settings
- `configs/perturbation/gaussian_blur.yaml`: perturbation dataset path and blur radius
- `configs/uncertainty/semantic_clustering.yaml`: NLI model, NLI batch size, entailment clustering settings

## Outputs

Inference JSON contains:

- `predictions`: sampled answer strings
- `prediction_details`: answer text, log probability, token length, normalized log probability

Semantic clustering JSON adds:

- `semantic_clusters`
- `cluster_texts`
- `num_semantic_clusters`

Semantic Entropy JSON contains:

- `id`
- `prompt`
- `images`
- `predictions`
- `sementic_entropy`

Reports contain:

- `roc_f1_summary.json`
- `roc_curve.csv`
- `roc_curve.png`

Current report directories:

- `reports/docvqa`
- `reports/docvqa_small`
- `reports/docvqa_high_temp`

## Implemented

- Qwen2.5-VL single and batch inference
- Multi-sample generation with `n_generation`
- Per-answer generation log probability extraction
- VizWiz, DocVQA, TextVQA dataset preparation
- Arrow dataset loading and saving
- Gaussian blur perturbation dataset generation
- NLI-based semantic equivalence clustering
- Semantic Entropy calculation from cluster probabilities
- Clean vs perturbed ROC-AUC and best F1 evaluation
- Matplotlib ROC curve generation with AUC annotation

## Notes

- `reports/` is intentionally tracked and should be committed with experiment results.
- `data/`, `outputs/`, and checkpoints remain ignored because they can be large or easily regenerated.
- The output key is currently spelled `sementic_entropy` for compatibility with existing generated files.
