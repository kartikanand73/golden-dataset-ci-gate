"""Metric calculations over prediction records."""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from typing import Sequence

from app.schemas import CATEGORIES, PredictionRecord

INVALID_LABEL = "__INVALID__"
SCHEMA_ERROR_TYPES = {"invalid_json", "schema_error"}


def per_label_scores(
    y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support per label. A None prediction is always wrong."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    preds = [p if p is not None else INVALID_LABEL for p in y_pred]
    scores: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, preds) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, preds) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, preds) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return scores


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]) -> float:
    scores = per_label_scores(y_true, y_pred, labels)
    return sum(s["f1"] for s in scores.values()) / len(labels)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return float(ordered[rank - 1])


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _round(value: float) -> float:
    return round(value, 6)


def compute_summary(records: Sequence[PredictionRecord]) -> dict:
    total = len(records)
    if total == 0:
        raise ValueError("Cannot summarise an empty run")

    y_true = [r.expected.category.value for r in records]
    y_pred = [r.candidate.category.value if r.candidate else None for r in records]
    scores = per_label_scores(y_true, y_pred, CATEGORIES)

    errors = Counter(r.error_type for r in records if r.error_type)
    latencies = [r.latency_ms for r in records]

    costs = [r.estimated_cost_usd for r in records]
    pricing_known = all(c is not None for c in costs)
    total_cost = sum((Decimal(c) for c in costs if c is not None), Decimal(0))
    cost_per_1000 = (total_cost / Decimal(total) * 1000) if pricing_known else None

    return {
        "total_cases": total,
        "category_accuracy": _round(sum(r.category_correct for r in records) / total),
        "macro_f1": _round(sum(s["f1"] for s in scores.values()) / len(CATEGORIES)),
        "human_review_accuracy": _round(sum(r.human_correct for r in records) / total),
        "fully_correct_rate": _round(sum(r.correct for r in records) / total),
        "critical_safety_errors": sum(r.critical_safety_error for r in records),
        "schema_failures": sum(n for t, n in errors.items() if t in SCHEMA_ERROR_TYPES),
        "errors_by_type": dict(sorted(errors.items())),
        "successful_call_rate": _round(sum(1 for r in records if r.candidate) / total),
        "total_retries": sum(r.retry_count for r in records),
        "median_latency_ms": _round(median(latencies)),
        "p95_latency_ms": _round(percentile(latencies, 95)),
        "input_tokens": sum(r.input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "pricing_known": pricing_known,
        "total_cost_usd": str(total_cost.quantize(Decimal("0.00000001"))),
        "cost_per_1000_tickets_usd": (
            str(cost_per_1000.quantize(Decimal("0.000001"))) if cost_per_1000 is not None else None
        ),
        "per_category": {
            label: {k: (_round(v) if isinstance(v, float) else v) for k, v in s.items()}
            for label, s in scores.items()
        },
    }
