"""NLI-based semantic equivalence clustering for generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from tqdm import tqdm


QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*Answer:", flags=re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class SemanticClusteringConfig:
    model_id: str = "cross-encoder/nli-deberta-v3-large"
    device: str | None = None
    batch_size: int = 32
    max_length: int = 512
    entailment_threshold: float | None = None
    premise_template: str = "Question: {question}\nAnswer: {answer}"
    hypothesis_template: str = "Question: {question}\nAnswer: {answer}"
    include_pair_scores: bool = False


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def clusters(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = {}
        for index in range(len(self.parent)):
            grouped.setdefault(self.find(index), []).append(index)
        return list(grouped.values())


def extract_question(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or row.get("question") or "")
    match = QUESTION_RE.search(prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def extract_prediction_texts(row: dict[str, Any]) -> list[str]:
    details = row.get("prediction_details")
    if isinstance(details, list):
        texts = [str(item.get("text", "")).strip() for item in details if isinstance(item, dict)]
    else:
        predictions = row.get("predictions", [])
        texts = [str(prediction).strip() for prediction in predictions]
    return [text for text in texts if text]


def format_nli_text(template: str, *, question: str, answer: str) -> str:
    return template.format(question=question, answer=answer)


class NLIEntailmentScorer:
    def __init__(self, config: SemanticClusteringConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Semantic clustering requires torch and transformers. Run `uv sync` first.") from exc

        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_id)
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.entailment_index = self._find_entailment_index()

    def _find_entailment_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for index, label in id2label.items():
            if str(label).lower() == "entailment":
                return int(index)
        # Model card label order: contradiction, entailment, neutral.
        return 1

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not pairs:
            return []
        scores: list[dict[str, Any]] = []
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            premises = [pair[0] for pair in batch]
            hypotheses = [pair[1] for pair in batch]
            features = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**features).logits
                probabilities = self.torch.softmax(logits, dim=-1)
                labels = probabilities.argmax(dim=-1)
            entailment_probs = probabilities[:, self.entailment_index].detach().cpu().tolist()
            label_ids = labels.detach().cpu().tolist()
            for entailment_prob, label_id in zip(entailment_probs, label_ids):
                scores.append(
                    {
                        "entailment_prob": float(entailment_prob),
                        "label_id": int(label_id),
                        "is_entailment": self._is_entailment(int(label_id), float(entailment_prob)),
                    }
                )
        return scores

    def _is_entailment(self, label_id: int, entailment_prob: float) -> bool:
        if self.config.entailment_threshold is not None:
            return entailment_prob >= self.config.entailment_threshold
        return label_id == self.entailment_index


def build_ordered_pairs(
    answers: list[str],
    *,
    question: str,
    premise_template: str,
    hypothesis_template: str,
) -> tuple[list[tuple[int, int]], list[tuple[str, str]]]:
    indices: list[tuple[int, int]] = []
    pairs: list[tuple[str, str]] = []
    for left_index, left_answer in enumerate(answers):
        for right_index, right_answer in enumerate(answers):
            if left_index == right_index:
                continue
            indices.append((left_index, right_index))
            pairs.append(
                (
                    format_nli_text(premise_template, question=question, answer=left_answer),
                    format_nli_text(hypothesis_template, question=question, answer=right_answer),
                )
            )
    return indices, pairs


def cluster_answers(
    *,
    answers: list[str],
    question: str,
    scorer: NLIEntailmentScorer,
    config: SemanticClusteringConfig,
) -> dict[str, Any]:
    if len(answers) <= 1:
        return {
            "semantic_clusters": [[index] for index in range(len(answers))],
            "cluster_texts": [[answer] for answer in answers],
            "num_semantic_clusters": len(answers),
            "nli_pairs": [] if config.include_pair_scores else None,
        }

    pair_indices, pair_texts = build_ordered_pairs(
        answers,
        question=question,
        premise_template=config.premise_template,
        hypothesis_template=config.hypothesis_template,
    )
    pair_scores = scorer.score_pairs(pair_texts)
    entailment_by_pair = {indices: score for indices, score in zip(pair_indices, pair_scores)}

    union_find = UnionFind(len(answers))
    for left_index in range(len(answers)):
        for right_index in range(left_index + 1, len(answers)):
            left_to_right = entailment_by_pair[(left_index, right_index)]["is_entailment"]
            right_to_left = entailment_by_pair[(right_index, left_index)]["is_entailment"]
            if left_to_right and right_to_left:
                union_find.union(left_index, right_index)

    clusters = union_find.clusters()
    result: dict[str, Any] = {
        "semantic_clusters": clusters,
        "cluster_texts": [[answers[index] for index in cluster] for cluster in clusters],
        "num_semantic_clusters": len(clusters),
    }
    if config.include_pair_scores:
        result["nli_pairs"] = [
            {"source": left, "target": right, **score}
            for (left, right), score in entailment_by_pair.items()
        ]
    return result


def _cluster_from_entailment_scores(
    *,
    answers: list[str],
    entailment_by_pair: dict[tuple[int, int], dict[str, Any]],
    include_pair_scores: bool,
) -> dict[str, Any]:
    if len(answers) <= 1:
        return {
            "semantic_clusters": [[index] for index in range(len(answers))],
            "cluster_texts": [[answer] for answer in answers],
            "num_semantic_clusters": len(answers),
        }

    union_find = UnionFind(len(answers))
    for left_index in range(len(answers)):
        for right_index in range(left_index + 1, len(answers)):
            left_to_right = entailment_by_pair[(left_index, right_index)]["is_entailment"]
            right_to_left = entailment_by_pair[(right_index, left_index)]["is_entailment"]
            if left_to_right and right_to_left:
                union_find.union(left_index, right_index)

    clusters = union_find.clusters()
    result: dict[str, Any] = {
        "semantic_clusters": clusters,
        "cluster_texts": [[answers[index] for index in cluster] for cluster in clusters],
        "num_semantic_clusters": len(clusters),
    }
    if include_pair_scores:
        result["nli_pairs"] = [
            {"source": left, "target": right, **score}
            for (left, right), score in entailment_by_pair.items()
        ]
    return result


def cluster_prediction_rows(
    rows: Iterable[dict[str, Any]],
    *,
    config: SemanticClusteringConfig,
) -> list[dict[str, Any]]:
    scorer = NLIEntailmentScorer(config)
    row_list = list(rows)
    row_answers = [extract_prediction_texts(row) for row in row_list]
    row_questions = [extract_question(row) for row in row_list]

    pair_tasks: list[tuple[int, int, int]] = []
    pair_texts: list[tuple[str, str]] = []
    for row_index, (answers, question) in enumerate(zip(row_answers, row_questions)):
        pair_indices, pairs = build_ordered_pairs(
            answers,
            question=question,
            premise_template=config.premise_template,
            hypothesis_template=config.hypothesis_template,
        )
        pair_tasks.extend((row_index, left, right) for left, right in pair_indices)
        pair_texts.extend(pairs)

    pair_scores = scorer.score_pairs(pair_texts)
    scores_by_row: list[dict[tuple[int, int], dict[str, Any]]] = [dict() for _ in row_list]
    for (row_index, left, right), score in zip(pair_tasks, pair_scores):
        scores_by_row[row_index][(left, right)] = score

    clustered_rows: list[dict[str, Any]] = []
    for row, answers, entailment_by_pair in tqdm(
        list(zip(row_list, row_answers, scores_by_row)),
        desc="semantic clustering",
    ):
        clustering = _cluster_from_entailment_scores(
            answers=answers,
            entailment_by_pair=entailment_by_pair,
            include_pair_scores=config.include_pair_scores,
        )
        output_row = dict(row)
        output_row.update(clustering)
        clustered_rows.append(output_row)
    return clustered_rows
