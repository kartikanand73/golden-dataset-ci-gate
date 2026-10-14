"""Run a model configuration against a dataset and write a reproducible run record."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app import __version__
from app.hashing import sha256_file, sha256_json
from app.metrics import compute_summary
from app.pricing import ModelPrice, estimate_cost, load_pricing
from app.providers import Provider, ProviderRequest, ProviderResult, build_provider
from app.safety import is_critical_safety_error, safety_result
from app.schemas import GoldenCase, ModelConfig, PredictionRecord, load_dataset, load_model_config

SUMMARY_FILE = "summary.json"
PREDICTIONS_FILE = "predictions.jsonl"
RUN_MANIFEST_FILE = "run-manifest.json"
RAW_RESPONSE_LIMIT = 2000


@dataclass
class EvaluationRun:
    output_dir: Path
    summary: dict
    manifest: dict
    records: list[PredictionRecord]


def git_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def resolve_config(config_path: str | Path, model_override: str | None = None) -> ModelConfig:
    config = load_model_config(config_path)
    if model_override:
        config = config.model_copy(update={"model": model_override})
    return config


def evaluate_case(
    case: GoldenCase,
    provider: Provider,
    config: ModelConfig,
    prompt: str,
    price: ModelPrice | None,
) -> PredictionRecord:
    request = ProviderRequest(
        model=config.model,
        prompt=prompt,
        ticket=case.ticket,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    try:
        result = provider.classify(request)
    except Exception as exc:  # providers should not raise; record it if one does
        result = ProviderResult(None, None, 0, 0, 0.0, 0, None,
                                "provider_exception", type(exc).__name__)

    output = result.output
    category_ok = output is not None and output.category == case.expected.category
    human_ok = output is not None and output.requires_human == case.expected.requires_human
    cost = estimate_cost(price, result.input_tokens, result.output_tokens)
    raw = result.raw_response[:RAW_RESPONSE_LIMIT] if result.raw_response else None

    return PredictionRecord(
        case_id=case.case_id,
        ticket=case.ticket,
        tags=case.tags,
        expected=case.expected,
        candidate=output,
        category_correct=category_ok,
        human_correct=human_ok,
        correct=category_ok and human_ok,
        safety_result=safety_result(case.expected, output),
        critical_safety_error=is_critical_safety_error(case.expected, output),
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost_usd=str(cost) if cost is not None else None,
        retry_count=result.retry_count,
        error_type=result.error_type,
        error_message=result.error_message,
        provider=config.provider,
        model_requested=config.model,
        model_served=result.model_served,
        raw_response=raw,
    )


def build_manifest(config: ModelConfig, dataset_path: Path, prompt_path: Path,
                   pricing_path: Path, summary: dict) -> dict:
    return {
        "tool_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset_path": dataset_path.as_posix(),
        "dataset_sha256": sha256_file(dataset_path),
        "provider": config.provider,
        "model": config.model,
        "prompt_sha256": sha256_file(prompt_path),
        "config_sha256": sha256_json(config.model_dump(mode="json")),
        "pricing_sha256": sha256_file(pricing_path) if pricing_path.exists() else None,
        "config": config.model_dump(mode="json"),
        "python_version": platform.python_version(),
        "metrics": {
            key: summary[key]
            for key in (
                "category_accuracy", "macro_f1", "human_review_accuracy",
                "critical_safety_errors", "schema_failures", "successful_call_rate",
                "p95_latency_ms", "cost_per_1000_tickets_usd",
            )
        },
    }


def write_run(output_dir: Path, summary: dict, records: list[PredictionRecord],
              manifest: dict, manifest_name: str = RUN_MANIFEST_FILE) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / PREDICTIONS_FILE).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_evaluation(
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_override: str | None = None,
    provider: Provider | None = None,
) -> EvaluationRun:
    config_path, dataset_path, output_dir = Path(config_path), Path(dataset_path), Path(output_dir)
    config = resolve_config(config_path, model_override)
    prompt_path = config_path.parent / config.prompt_file
    pricing_path = config_path.parent / config.pricing_file
    prompt = prompt_path.read_text(encoding="utf-8")
    price = load_pricing(pricing_path).get(config.model)
    cases = load_dataset(dataset_path)
    provider = provider or build_provider(config)

    records = [evaluate_case(case, provider, config, prompt, price) for case in cases]
    summary = compute_summary(records)
    manifest = build_manifest(config, dataset_path, prompt_path, pricing_path, summary)
    write_run(output_dir, summary, records, manifest)
    return EvaluationRun(output_dir, summary, manifest, records)
