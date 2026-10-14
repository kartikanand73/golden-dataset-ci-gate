import json

from app.comparator import compare, load_run, render_markdown, write_comparison
from app.schemas import load_gate_policy
from tests.conftest import GATES


def test_changed_cases_appear_in_comparison_report(baseline_dir, mock_runs, tmp_path):
    comparison = compare(load_run(baseline_dir), load_run(mock_runs["mock-v2-regression"]),
                         load_gate_policy(GATES))
    changed_ids = {c.case_id for c in comparison.changed_cases}
    assert {"case-031", "case-032", "case-034", "case-039", "case-045"} <= changed_ids

    out = tmp_path / "comparison"
    write_comparison(comparison, out)
    markdown = (out / "comparison.md").read_text()
    for case_id in changed_ids:
        assert case_id in markdown
    assert "CRITICAL SAFETY ERROR" in markdown

    lines = (out / "changed-cases.jsonl").read_text().splitlines()
    assert {json.loads(l)["case_id"] for l in lines} == changed_ids
    assert all((out / name).exists() for name in ("comparison.json", "junit.xml"))


def test_regressions_and_new_failures_by_category(baseline_dir, mock_runs):
    comparison = compare(load_run(baseline_dir), load_run(mock_runs["mock-v2-regression"]),
                         load_gate_policy(GATES))
    assert len(comparison.regressions) == 5
    assert comparison.new_failures_by_category == {"SECURITY_INCIDENT": 5}
    critical = [c.case_id for c in comparison.changed_cases if c.candidate_critical]
    assert critical == ["case-032", "case-034", "case-039", "case-045"]


def test_improvements_are_reported(baseline_dir, mock_runs):
    comparison = compare(load_run(baseline_dir), load_run(mock_runs["mock-v3-fixed"]),
                         load_gate_policy(GATES))
    assert {c.case_id for c in comparison.improvements} == {"case-041", "case-054"}
    assert comparison.regressions == []


def test_identical_runs_have_no_changes(baseline_dir, mock_runs):
    comparison = compare(load_run(baseline_dir), load_run(mock_runs["mock-v1-baseline"]),
                         load_gate_policy(GATES))
    assert comparison.passed
    assert comparison.changed_cases == []
    assert "No routing decision changed" in render_markdown(comparison)
