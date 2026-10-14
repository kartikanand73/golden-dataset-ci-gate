import json
import shutil

import pytest

from app.baseline import BaselineUpdateNotAllowed, approve_baseline
from app.change_scope import MOVING_GOALPOSTS_MESSAGE, classify_changes
from app.cli import main
from app.comparator import FAIL, PASS, WARN, compare, load_run, render_markdown
from app.schemas import load_gate_policy
from tests.conftest import GATES, GOLDEN


def _gate(comparison, name):
    return next(g for g in comparison.gates if g.name == name)


# ---------------------------------------------------------------- gate outcomes


def test_candidate_regression_fails_gate(baseline_dir, mock_runs):
    comparison = compare(load_run(baseline_dir), load_run(mock_runs["mock-v2-regression"]),
                         load_gate_policy(GATES))
    assert not comparison.passed
    failed = {g.name for g in comparison.failed_gates}
    assert {"Critical safety errors", "Category accuracy", "Accuracy regression"} <= failed
    assert render_markdown(comparison).startswith("GOLDEN DATASET GATE: FAIL")


def test_candidate_improvement_passes_gate(baseline_dir, mock_runs):
    comparison = compare(load_run(baseline_dir), load_run(mock_runs["mock-v3-fixed"]),
                         load_gate_policy(GATES))
    assert comparison.passed
    assert render_markdown(comparison).startswith("GOLDEN DATASET GATE: PASS")


def test_cost_and_latency_increases_create_warnings_not_failures(baseline_dir, tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(baseline_dir, candidate)
    summary = json.loads((candidate / "summary.json").read_text())
    summary["p95_latency_ms"] *= 1.5
    summary["cost_per_1000_tickets_usd"] = str(float(summary["cost_per_1000_tickets_usd"]) * 1.5)
    (candidate / "summary.json").write_text(json.dumps(summary))

    comparison = compare(load_run(baseline_dir), load_run(candidate), load_gate_policy(GATES))
    assert comparison.passed
    assert _gate(comparison, "P95 latency increase").status == WARN
    assert _gate(comparison, "Cost per 1,000 tickets increase").status == WARN
    assert len(comparison.warnings) == 2


def test_dataset_mismatch_fails_gate(baseline_dir, mock_runs, tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(mock_runs["mock-v1-baseline"], candidate)
    manifest = json.loads((candidate / "run-manifest.json").read_text())
    manifest["dataset_sha256"] = "0" * 64
    (candidate / "run-manifest.json").write_text(json.dumps(manifest))
    comparison = compare(load_run(baseline_dir), load_run(candidate), load_gate_policy(GATES))
    assert _gate(comparison, "Same golden dataset").status == FAIL
    assert not comparison.passed


def test_gate_cli_exit_codes_and_reports(baseline_dir, mock_runs, tmp_path):
    for model, expected_code in (("mock-v1-baseline", 0), ("mock-v2-regression", 1), ("mock-v3-fixed", 0)):
        out = tmp_path / model
        code = main(["gate", "--baseline", str(baseline_dir), "--candidate", str(mock_runs[model]),
                     "--policy", str(GATES), "--output", str(out)])
        assert code == expected_code
        # Reports exist even when the gate fails.
        for name in ("comparison.json", "comparison.md", "changed-cases.jsonl", "junit.xml"):
            assert (out / name).exists()
        result = json.loads((out / "comparison.json").read_text())["result"]
        assert result == (PASS if expected_code == 0 else FAIL)


# ---------------------------------------------------------------- baseline protection


def test_baseline_cannot_be_overwritten_without_flag(baseline_dir, mock_runs):
    before = {p.name: p.read_bytes() for p in baseline_dir.iterdir()}
    with pytest.raises(BaselineUpdateNotAllowed):
        approve_baseline(mock_runs["mock-v2-regression"], baseline_dir, GOLDEN, env={})
    with pytest.raises(BaselineUpdateNotAllowed):
        approve_baseline(mock_runs["mock-v2-regression"], baseline_dir, GOLDEN,
                         env={"ALLOW_BASELINE_UPDATE": "yes"})
    assert {p.name: p.read_bytes() for p in baseline_dir.iterdir()} == before


def test_baseline_cli_refuses_without_flag(baseline_dir, mock_runs, monkeypatch):
    monkeypatch.delenv("ALLOW_BASELINE_UPDATE", raising=False)
    code = main(["approve-baseline", "--source", str(mock_runs["mock-v3-fixed"]),
                 "--baseline-dir", str(baseline_dir), "--dataset", str(GOLDEN)])
    assert code == 3
    assert json.loads((baseline_dir / "manifest.json").read_text())["model"] == "mock-v1-baseline"


def test_baseline_can_be_approved_explicitly(baseline_dir, mock_runs):
    manifest = approve_baseline(mock_runs["mock-v3-fixed"], baseline_dir, GOLDEN,
                                env={"ALLOW_BASELINE_UPDATE": "true", "USER": "reviewer"})
    assert manifest["model"] == "mock-v3-fixed"
    assert manifest["approval"]["approved_by"] == "reviewer"
    stored = json.loads((baseline_dir / "manifest.json").read_text())
    for key in ("dataset_sha256", "provider", "model", "prompt_sha256", "config_sha256",
                "timestamp", "git_commit", "metrics"):
        assert key in stored


# ---------------------------------------------------------------- moving the goalposts


@pytest.mark.parametrize("system_file", [
    "config/model.yaml", "config/prompt.txt", "app/providers/mock.py",
    "app/metrics.py", "app/evaluator.py", "config/gates.yaml",
])
def test_dataset_and_system_changes_together_are_detected(system_file):
    scope = classify_changes(["data/golden_dataset.jsonl", system_file, "README.md"])
    assert scope.violation


def test_separate_changes_are_allowed():
    assert not classify_changes(["data/golden_dataset.jsonl", "README.md"]).violation
    assert not classify_changes(["config/model.yaml", "config/prompt.txt"]).violation
    assert not classify_changes(["data/development_dataset.jsonl", "config/prompt.txt"]).violation


def test_change_scope_cli_message(capsys):
    code = main(["check-change-scope", "--files", "data/golden_dataset.jsonl", "config/model.yaml"])
    assert code == 1
    assert MOVING_GOALPOSTS_MESSAGE in capsys.readouterr().err
