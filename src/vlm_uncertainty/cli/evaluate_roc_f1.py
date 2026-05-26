"""CLI for ROC/F1 evaluation of clean vs perturbed semantic entropy."""

from __future__ import annotations

import argparse

from vlm_uncertainty.evaluation.roc_f1 import evaluate_clean_vs_perturbed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean vs perturbed semantic entropy with ROC and F1.")
    parser.add_argument("--clean", default='outputs/clean/predictions_semantic_entropy.json', help="Clean dataset semantic entropy JSON path.")
    parser.add_argument("--perturbed", default='outputs/perturbed/predictions_semantic_entropy.json', help="Perturbed dataset semantic entropy JSON path.")
    parser.add_argument("--report-dir", default="reports", help="Directory to write report artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_clean_vs_perturbed(
        clean_json=args.clean,
        perturbed_json=args.perturbed,
        report_dir=args.report_dir,
    )
    print(f"AUC: {summary['auc']:.6f}")
    print(f"Best F1: {summary['best_f1']['f1']:.6f} @ threshold {summary['best_f1']['threshold']}")
    print(f"Reports written to {args.report_dir}")


if __name__ == "__main__":
    main()
