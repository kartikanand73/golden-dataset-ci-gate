from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.evaluator import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_dataset.jsonl"
CONFIG = ROOT / "config" / "model.yaml"
GATES = ROOT / "config" / "gates.yaml"


@pytest.fixture(scope="session")
def mock_runs(tmp_path_factory) -> dict[str, Path]:
    """Evaluate each demo mock once against the real golden dataset."""
    base = tmp_path_factory.mktemp("runs")
    runs = {}
    for model in ("mock-v1-baseline", "mock-v2-regression", "mock-v3-fixed"):
        out = base / model
        run_evaluation(CONFIG, GOLDEN, out, model_override=model)
        runs[model] = out
    return runs


@pytest.fixture
def baseline_dir(mock_runs, tmp_path) -> Path:
    """A baseline directory built from mock-v1 (manifest.json naming)."""
    target = tmp_path / "baselines"
    target.mkdir()
    source = mock_runs["mock-v1-baseline"]
    shutil.copyfile(source / "summary.json", target / "summary.json")
    shutil.copyfile(source / "predictions.jsonl", target / "predictions.jsonl")
    shutil.copyfile(source / "run-manifest.json", target / "manifest.json")
    return target


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


def case(case_id: str, category: str = "PASSWORD_RESET", human: bool = False) -> dict:
    return {"case_id": case_id, "ticket": f"ticket {case_id}",
            "expected": {"category": category, "requires_human": human}, "tags": []}
