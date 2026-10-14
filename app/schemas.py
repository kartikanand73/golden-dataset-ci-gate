"""Typed schemas for datasets, model outputs, configuration and predictions."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError


class Category(str, Enum):
    PASSWORD_RESET = "PASSWORD_RESET"
    ACCESS_REQUEST = "ACCESS_REQUEST"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"
    GENERAL_SUPPORT = "GENERAL_SUPPORT"


CATEGORIES: tuple[str, ...] = tuple(c.value for c in Category)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- datasets


class ExpectedOutput(_Strict):
    category: Category
    requires_human: StrictBool


class GoldenCase(_Strict):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    ticket: str = Field(min_length=1)
    expected: ExpectedOutput
    tags: list[str] = Field(default_factory=list)


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(p) for p in item["loc"]) or "<root>"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def load_dataset(path: str | Path) -> list[GoldenCase]:
    """Load and validate a JSONL dataset. Rejects invalid labels and duplicate IDs."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    errors: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        try:
            case = GoldenCase.model_validate(raw)
        except ValidationError as exc:
            errors.append(f"line {line_no}: {_format_validation_error(exc)}")
            continue
        if case.case_id in seen:
            errors.append(f"line {line_no}: duplicate case_id '{case.case_id}'")
            continue
        seen.add(case.case_id)
        cases.append(case)

    if errors:
        raise DatasetError(f"{path} failed validation:\n  " + "\n  ".join(errors))
    if not cases:
        raise DatasetError(f"{path} contains no cases")
    return cases


def category_counts(cases: list[GoldenCase]) -> dict[str, int]:
    counts = {c: 0 for c in CATEGORIES}
    for case in cases:
        counts[case.expected.category.value] += 1
    return counts


# --------------------------------------------------------------------------- model output


class ModelOutput(_Strict):
    """The fixed output schema every provider must satisfy."""

    category: Category
    requires_human: StrictBool
    confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------- configuration

ProviderName = Literal["mock", "openai_compatible", "anthropic"]


class ModelConfig(_Strict):
    provider: ProviderName
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    # Paths are resolved relative to the directory containing the model config.
    prompt_file: str = "prompt.txt"
    pricing_file: str = "pricing.yaml"


class HardGates(_Strict):
    minimum_category_accuracy: float = Field(ge=0, le=1)
    minimum_macro_f1: float = Field(ge=0, le=1)
    minimum_human_review_accuracy: float = Field(ge=0, le=1)
    maximum_critical_safety_errors: int = Field(ge=0)
    maximum_accuracy_regression_points: float = Field(ge=0)
    maximum_macro_f1_regression_points: float = Field(ge=0)


class WarningGates(_Strict):
    maximum_p95_latency_increase_percent: float = Field(ge=0)
    maximum_cost_increase_percent: float = Field(ge=0)


class GatePolicy(_Strict):
    hard_gates: HardGates
    warning_gates: WarningGates


def _load_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml(path))


def load_gate_policy(path: str | Path) -> GatePolicy:
    return GatePolicy.model_validate(_load_yaml(path))


# --------------------------------------------------------------------------- predictions


class PredictionRecord(_Strict):
    case_id: str
    ticket: str
    tags: list[str]
    expected: ExpectedOutput
    candidate: ModelOutput | None
    category_correct: bool
    human_correct: bool
    correct: bool
    safety_result: str
    critical_safety_error: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: str | None
    retry_count: int
    error_type: str | None
    error_message: str | None
    provider: str
    model_requested: str
    model_served: str | None
    raw_response: str | None


def read_predictions(path: str | Path) -> list[PredictionRecord]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PredictionRecord.model_validate_json(line))
    return records
