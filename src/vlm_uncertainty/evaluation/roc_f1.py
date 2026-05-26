"""ROC and F1 evaluation for clean vs perturbed semantic entropy scores."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_uncertainty.utils.io import read_json, write_json


@dataclass(frozen=True)
class ScoredExample:
    id: str
    label: int
    score: float
    source: str


def _entropy(row: dict[str, Any]) -> float:
    value = row.get("sementic_entropy", row.get("semantic_entropy"))
    if value is None:
        raise ValueError(f"Missing semantic entropy for row id={row.get('id')}")
    return float(value)


def load_scored_examples(clean_path: str | Path, perturbed_path: str | Path) -> list[ScoredExample]:
    clean_rows = read_json(clean_path)
    perturbed_rows = read_json(perturbed_path)
    if not isinstance(clean_rows, list):
        raise ValueError(f"Clean JSON must contain a list: {clean_path}")
    if not isinstance(perturbed_rows, list):
        raise ValueError(f"Perturbed JSON must contain a list: {perturbed_path}")

    examples: list[ScoredExample] = []
    for row in clean_rows:
        examples.append(ScoredExample(id=str(row.get("id")), label=0, score=_entropy(row), source="clean"))
    for row in perturbed_rows:
        examples.append(ScoredExample(id=str(row.get("id")), label=1, score=_entropy(row), source="perturbed"))
    return examples


def _best_f1(labels: list[int], scores: list[float]) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support

    thresholds = sorted(set(scores), reverse=True)
    thresholds = [float("inf"), *thresholds, float("-inf")]
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        preds = [int(score >= threshold) for score in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
        )
        tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, preds))
        fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, preds))
        tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, preds))
        fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, preds))
        candidate = {
            "threshold": threshold,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        if best is None or (candidate["f1"], candidate["recall"], -candidate["fp"]) > (
            best["f1"],
            best["recall"],
            -best["fp"],
        ):
            best = candidate
    if best is None:
        raise ValueError("No examples available for F1 evaluation")
    return best


def _write_roc_csv(path: str | Path, fpr: list[float], tpr: list[float], thresholds: list[float]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "fpr", "tpr"])
        writer.writeheader()
        for threshold, x_value, y_value in zip(thresholds, fpr, tpr):
            writer.writerow({"threshold": threshold, "fpr": x_value, "tpr": y_value})


def _write_roc_plot(path: str | Path, fpr: list[float], tpr: list[float], auc_value: float, best: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label=f"ROC curve (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linewidth=1.5, linestyle="--", label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2563eb")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Clean vs Perturbed Detection by Semantic Entropy", fontsize=14, pad=14)
    ax.text(
        0.98,
        0.04,
        f"Best F1 = {best['f1']:.3f}\nThreshold = {best['threshold']:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_clean_vs_perturbed(
    *,
    clean_json: str | Path,
    perturbed_json: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    from sklearn.metrics import auc, roc_curve

    examples = load_scored_examples(clean_json, perturbed_json)
    labels = [example.label for example in examples]
    scores = [example.score for example in examples]
    if len(set(labels)) != 2:
        raise ValueError("ROC/F1 evaluation requires both clean and perturbed examples")

    fpr_array, tpr_array, threshold_array = roc_curve(labels, scores, pos_label=1)
    auc_value = float(auc(fpr_array, tpr_array))
    best = _best_f1(labels, scores)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    fpr = [float(value) for value in fpr_array]
    tpr = [float(value) for value in tpr_array]
    thresholds = [float(value) for value in threshold_array]
    _write_roc_csv(report_path / "roc_curve.csv", fpr, tpr, thresholds)
    _write_roc_plot(report_path / "roc_curve.png", fpr, tpr, auc_value, best)

    summary = {
        "clean_json": str(clean_json),
        "perturbed_json": str(perturbed_json),
        "n_clean": sum(example.label == 0 for example in examples),
        "n_perturbed": sum(example.label == 1 for example in examples),
        "score_key": "sementic_entropy",
        "positive_label": "perturbed",
        "auc": auc_value,
        "best_f1": best,
        "roc_curve_png": str(report_path / "roc_curve.png"),
        "roc_curve_csv": str(report_path / "roc_curve.csv"),
    }
    write_json(report_path / "roc_f1_summary.json", summary)
    return summary
