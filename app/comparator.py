"""Compare a candidate run with the approved baseline and apply the gate policy."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from app.schemas import GatePolicy, PredictionRecord, read_predictions

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


# --------------------------------------------------------------------------- loading


@dataclass
class RunData:
    summary: dict
    manifest: dict
    predictions: dict[str, PredictionRecord]


def load_run(directory: str | Path) -> RunData:
    directory = Path(directory)
    manifest_path = next(
        (directory / n for n in ("manifest.json", "run-manifest.json") if (directory / n).exists()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(f"No manifest.json or run-manifest.json in {directory}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = {r.case_id: r for r in read_predictions(directory / "predictions.jsonl")}
    return RunData(summary, manifest, predictions)


# --------------------------------------------------------------------------- results


@dataclass
class GateResult:
    name: str
    kind: str  # "hard" or "warning"
    status: str  # PASS / FAIL / WARN
    baseline: str
    candidate: str
    threshold: str
    message: str


@dataclass
class ChangedCase:
    case_id: str
    ticket: str
    expected_category: str
    expected_requires_human: bool
    baseline_output: dict | None
    candidate_output: dict | None
    baseline_correct: bool
    candidate_correct: bool
    candidate_critical: bool
    change_type: str  # REGRESSION / IMPROVEMENT / CHANGED_STILL_WRONG
    baseline_error: str | None
    candidate_error: str | None


@dataclass
class Comparison:
    passed: bool
    baseline_model: str
    candidate_model: str
    dataset_sha256: str
    gates: list[GateResult]
    metrics: list[dict]
    changed_cases: list[ChangedCase]
    new_failures_by_category: dict[str, int]
    improvements_by_category: dict[str, int]
    confidence: dict
    candidate_errors: list[dict]
    errors_by_type: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if g.status == FAIL]

    @property
    def regressions(self) -> list[ChangedCase]:
        return [c for c in self.changed_cases if c.change_type == "REGRESSION"]

    @property
    def improvements(self) -> list[ChangedCase]:
        return [c for c in self.changed_cases if c.change_type == "IMPROVEMENT"]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["result"] = PASS if self.passed else FAIL
        data["failed_gates"] = [g.name for g in self.failed_gates]
        data["regression_count"] = len(self.regressions)
        data["improvement_count"] = len(self.improvements)
        return data


# --------------------------------------------------------------------------- formatting helpers


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pts(delta: float) -> str:
    return f"{delta * 100:+.2f} pt"


def _ms(value: float) -> str:
    return f"{value:.0f} ms"


def _usd(value: str | None) -> str:
    return "n/a" if value is None else f"${Decimal(value):.4f}"


def _pct_change(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100


def _output_dict(record: PredictionRecord | None) -> dict | None:
    if record is None or record.candidate is None:
        return None
    return record.candidate.model_dump(mode="json")


def _decision(record: PredictionRecord | None) -> tuple | None:
    if record is None or record.candidate is None:
        return None
    return (record.candidate.category.value, record.candidate.requires_human)


# --------------------------------------------------------------------------- gates


def _evaluate_gates(base: dict, cand: dict, policy: GatePolicy,
                    same_dataset: bool, dataset_message: str) -> list[GateResult]:
    hard, warn = policy.hard_gates, policy.warning_gates
    gates = [
        GateResult("Same golden dataset", "hard", PASS if same_dataset else FAIL,
                   "-", "-", "identical SHA-256 and case IDs", dataset_message),
    ]

    def minimum(name: str, key: str, threshold: float) -> None:
        ok = cand[key] >= threshold
        gates.append(GateResult(
            name, "hard", PASS if ok else FAIL, _pct(base[key]), _pct(cand[key]),
            f">= {_pct(threshold)}",
            f"{_pct(cand[key])} {'meets' if ok else 'is below'} the {_pct(threshold)} minimum",
        ))

    def regression(name: str, key: str, max_points: float) -> None:
        drop = (base[key] - cand[key]) * 100
        ok = drop <= max_points + 1e-9
        gates.append(GateResult(
            name, "hard", PASS if ok else FAIL, _pct(base[key]), _pct(cand[key]),
            f"drop <= {max_points:.1f} pt",
            f"changed by {_pts(cand[key] - base[key])} versus baseline",
        ))

    minimum("Category accuracy", "category_accuracy", hard.minimum_category_accuracy)
    regression("Accuracy regression", "category_accuracy", hard.maximum_accuracy_regression_points)
    minimum("Macro-F1", "macro_f1", hard.minimum_macro_f1)
    regression("Macro-F1 regression", "macro_f1", hard.maximum_macro_f1_regression_points)
    minimum("Human-review accuracy", "human_review_accuracy", hard.minimum_human_review_accuracy)

    critical = cand["critical_safety_errors"]
    ok = critical <= hard.maximum_critical_safety_errors
    gates.append(GateResult(
        "Critical safety errors", "hard", PASS if ok else FAIL,
        str(base["critical_safety_errors"]), str(critical),
        f"<= {hard.maximum_critical_safety_errors}",
        f"{critical} security incident(s) routed without human review",
    ))

    change = _pct_change(base["p95_latency_ms"], cand["p95_latency_ms"])
    limit = warn.maximum_p95_latency_increase_percent
    status = WARN if change is not None and change > limit else PASS
    gates.append(GateResult(
        "P95 latency increase", "warning", status, _ms(base["p95_latency_ms"]),
        _ms(cand["p95_latency_ms"]), f"increase <= {limit:.0f}%",
        "baseline p95 is zero; not compared" if change is None else f"{change:+.1f}% versus baseline",
    ))

    base_cost, cand_cost = base.get("cost_per_1000_tickets_usd"), cand.get("cost_per_1000_tickets_usd")
    limit = warn.maximum_cost_increase_percent
    if base_cost is None or cand_cost is None or Decimal(base_cost) == 0:
        status, message = PASS, "pricing unknown for baseline or candidate; not compared"
    else:
        change = float((Decimal(cand_cost) - Decimal(base_cost)) / Decimal(base_cost) * 100)
        status = WARN if change > limit else PASS
        message = f"{change:+.1f}% versus baseline"
    gates.append(GateResult(
        "Cost per 1,000 tickets increase", "warning", status, _usd(base_cost), _usd(cand_cost),
        f"increase <= {limit:.0f}%", message,
    ))
    return gates


def _metric_rows(base: dict, cand: dict, gates: list[GateResult]) -> list[dict]:
    by_name = {g.name: g for g in gates}

    def result(*names: str) -> str:
        statuses = [by_name[n].status for n in names]
        return FAIL if FAIL in statuses else WARN if WARN in statuses else PASS

    def gate_text(*names: str) -> str:
        return "; ".join(by_name[n].threshold for n in names)

    cost_b, cost_c = base.get("cost_per_1000_tickets_usd"), cand.get("cost_per_1000_tickets_usd")
    cost_diff = "n/a" if cost_b is None or cost_c is None else (
        f"{Decimal(cost_c) - Decimal(cost_b):+.4f} USD")
    latency_change = _pct_change(base["p95_latency_ms"], cand["p95_latency_ms"])

    return [
        {"metric": "Accuracy", "baseline": _pct(base["category_accuracy"]),
         "candidate": _pct(cand["category_accuracy"]),
         "difference": _pts(cand["category_accuracy"] - base["category_accuracy"]),
         "gate": gate_text("Category accuracy", "Accuracy regression"),
         "result": result("Category accuracy", "Accuracy regression")},
        {"metric": "Macro-F1", "baseline": _pct(base["macro_f1"]), "candidate": _pct(cand["macro_f1"]),
         "difference": _pts(cand["macro_f1"] - base["macro_f1"]),
         "gate": gate_text("Macro-F1", "Macro-F1 regression"),
         "result": result("Macro-F1", "Macro-F1 regression")},
        {"metric": "Human-review accuracy", "baseline": _pct(base["human_review_accuracy"]),
         "candidate": _pct(cand["human_review_accuracy"]),
         "difference": _pts(cand["human_review_accuracy"] - base["human_review_accuracy"]),
         "gate": gate_text("Human-review accuracy"), "result": result("Human-review accuracy")},
        {"metric": "Critical safety errors", "baseline": str(base["critical_safety_errors"]),
         "candidate": str(cand["critical_safety_errors"]),
         "difference": f"{cand['critical_safety_errors'] - base['critical_safety_errors']:+d}",
         "gate": gate_text("Critical safety errors"), "result": result("Critical safety errors")},
        {"metric": "Invalid JSON / schema failures", "baseline": str(base["schema_failures"]),
         "candidate": str(cand["schema_failures"]),
         "difference": f"{cand['schema_failures'] - base['schema_failures']:+d}",
         "gate": "reported", "result": "INFO"},
        {"metric": "P95 latency", "baseline": _ms(base["p95_latency_ms"]),
         "candidate": _ms(cand["p95_latency_ms"]),
         "difference": "n/a" if latency_change is None else f"{latency_change:+.1f}%",
         "gate": gate_text("P95 latency increase"), "result": result("P95 latency increase")},
        {"metric": "Cost per 1,000", "baseline": _usd(cost_b), "candidate": _usd(cost_c),
         "difference": cost_diff, "gate": gate_text("Cost per 1,000 tickets increase"),
         "result": result("Cost per 1,000 tickets increase")},
    ]


# --------------------------------------------------------------------------- case diff


def _changed_cases(baseline: RunData, candidate: RunData) -> list[ChangedCase]:
    changed = []
    for case_id in sorted(set(baseline.predictions) | set(candidate.predictions)):
        b, c = baseline.predictions.get(case_id), candidate.predictions.get(case_id)
        b_error = b.error_type if b else "missing"
        c_error = c.error_type if c else "missing"
        if _decision(b) == _decision(c) and b_error == c_error:
            continue
        ref = c or b
        b_ok, c_ok = bool(b and b.correct), bool(c and c.correct)
        change_type = ("REGRESSION" if b_ok and not c_ok else
                       "IMPROVEMENT" if c_ok and not b_ok else "CHANGED_STILL_WRONG")
        changed.append(ChangedCase(
            case_id=case_id,
            ticket=ref.ticket,
            expected_category=ref.expected.category.value,
            expected_requires_human=ref.expected.requires_human,
            baseline_output=_output_dict(b),
            candidate_output=_output_dict(c),
            baseline_correct=b_ok,
            candidate_correct=c_ok,
            candidate_critical=bool(c and c.critical_safety_error),
            change_type=change_type,
            baseline_error=b_error,
            candidate_error=c_error,
        ))
    order = {"REGRESSION": 0, "IMPROVEMENT": 1, "CHANGED_STILL_WRONG": 2}
    return sorted(changed, key=lambda x: (order[x.change_type], not x.candidate_critical, x.case_id))


def _confidence_changes(baseline: RunData, candidate: RunData) -> dict:
    deltas = []
    for case_id, b in baseline.predictions.items():
        c = candidate.predictions.get(case_id)
        if b.candidate and c and c.candidate:
            deltas.append((case_id, c.candidate.confidence - b.candidate.confidence))
    if not deltas:
        return {"compared_cases": 0, "mean_change": 0.0, "max_abs_change": 0.0, "large_changes": []}
    large = [{"case_id": cid, "change": round(d, 4)} for cid, d in deltas if abs(d) >= 0.10]
    return {
        "compared_cases": len(deltas),
        "mean_change": round(sum(d for _, d in deltas) / len(deltas), 4),
        "max_abs_change": round(max(abs(d) for _, d in deltas), 4),
        "large_changes": large,
    }


# --------------------------------------------------------------------------- entry point


def compare(baseline: RunData, candidate: RunData, policy: GatePolicy) -> Comparison:
    b_hash, c_hash = baseline.manifest["dataset_sha256"], candidate.manifest["dataset_sha256"]
    same_ids = set(baseline.predictions) == set(candidate.predictions)
    same_dataset = b_hash == c_hash and same_ids
    if same_dataset:
        dataset_message = f"both runs used dataset {c_hash[:12]}"
    elif b_hash != c_hash:
        dataset_message = (f"baseline used {b_hash[:12]} but candidate used {c_hash[:12]}; "
                           "results are not comparable")
    else:
        dataset_message = "case IDs differ between baseline and candidate"

    gates = _evaluate_gates(baseline.summary, candidate.summary, policy, same_dataset, dataset_message)
    changed = _changed_cases(baseline, candidate)
    regressions = [c for c in changed if c.change_type == "REGRESSION"]
    improvements = [c for c in changed if c.change_type == "IMPROVEMENT"]
    errors = [
        {"case_id": r.case_id, "error_type": r.error_type, "error_message": r.error_message,
         "retry_count": r.retry_count}
        for r in candidate.predictions.values() if r.error_type
    ]
    return Comparison(
        passed=not any(g.status == FAIL for g in gates),
        baseline_model=f"{baseline.manifest['provider']}/{baseline.manifest['model']}",
        candidate_model=f"{candidate.manifest['provider']}/{candidate.manifest['model']}",
        dataset_sha256=c_hash,
        gates=gates,
        metrics=_metric_rows(baseline.summary, candidate.summary, gates),
        changed_cases=changed,
        new_failures_by_category=dict(sorted(Counter(c.expected_category for c in regressions).items())),
        improvements_by_category=dict(sorted(Counter(c.expected_category for c in improvements).items())),
        confidence=_confidence_changes(baseline, candidate),
        candidate_errors=errors,
        errors_by_type=candidate.summary.get("errors_by_type", {}),
        warnings=[f"{g.name}: {g.message}" for g in gates if g.status == WARN],
    )


# --------------------------------------------------------------------------- rendering


def _describe(output: dict | None, error: str | None) -> str:
    if output is None:
        return f"no valid output ({error or 'missing'})"
    human = "true" if output["requires_human"] else "false"
    return f"{output['category']} / requires_human={human} / confidence {output['confidence']:.2f}"


def render_markdown(comparison: Comparison) -> str:
    c = comparison
    lines = [f"GOLDEN DATASET GATE: {PASS if c.passed else FAIL}", ""]
    lines += [
        f"- **Baseline:** `{c.baseline_model}`",
        f"- **Candidate:** `{c.candidate_model}`",
        f"- **Golden dataset SHA-256:** `{c.dataset_sha256}`",
        f"- **Changed decisions:** {len(c.changed_cases)} "
        f"({len(c.regressions)} regressions, {len(c.improvements)} improvements)",
        "",
        "| Metric | Baseline | Candidate | Difference | Gate | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in c.metrics:
        lines.append(f"| {row['metric']} | {row['baseline']} | {row['candidate']} | "
                     f"{row['difference']} | {row['gate']} | {row['result']} |")

    lines += ["", "## Why the gate " + ("passed" if c.passed else "failed"), ""]
    if c.failed_gates:
        for gate in c.failed_gates:
            lines.append(f"- **FAIL: {gate.name}.** {gate.message} (gate: {gate.threshold}).")
    else:
        lines.append("- Every hard quality and safety gate passed.")
    for warning in c.warnings:
        lines.append(f"- **WARN: {warning}.** Warnings do not block deployment.")

    lines += ["", "## Changed cases", ""]
    if not c.changed_cases:
        lines.append("No routing decision changed between baseline and candidate.")
    for case in c.changed_cases:
        flag = " **CRITICAL SAFETY ERROR**" if case.candidate_critical else ""
        human = "true" if case.expected_requires_human else "false"
        lines += [
            f"- **{case.case_id}: {case.change_type}**{flag}",
            f"  - Ticket: \"{case.ticket}\"",
            f"  - Expected: {case.expected_category} / requires_human={human}",
            f"  - Baseline: {_describe(case.baseline_output, case.baseline_error)} "
            f"({'correct' if case.baseline_correct else 'wrong'})",
            f"  - Candidate: {_describe(case.candidate_output, case.candidate_error)} "
            f"({'correct' if case.candidate_correct else 'wrong'})",
        ]

    lines += ["", "## New failures by category", ""]
    if c.new_failures_by_category:
        lines += [f"- {cat}: {n}" for cat, n in c.new_failures_by_category.items()]
    else:
        lines.append("- None")
    if c.improvements_by_category:
        lines += ["", "## Improvements by category", ""]
        lines += [f"- {cat}: {n}" for cat, n in c.improvements_by_category.items()]

    conf = c.confidence
    lines += ["", "## Confidence changes", "",
              f"- Cases compared: {conf['compared_cases']}; mean change {conf['mean_change']:+.4f}; "
              f"largest absolute change {conf['max_abs_change']:.4f}",
              f"- Cases with a change of 0.10 or more: {len(conf['large_changes'])}"]

    lines += ["", "## Errors and timeouts", ""]
    if c.candidate_errors:
        for err in c.candidate_errors:
            lines.append(f"- {err['case_id']}: {err['error_type']} ({err['error_message']}), "
                         f"retries {err['retry_count']}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def render_junit(comparison: Comparison) -> str:
    failures = len(comparison.failed_gates)
    cases = []
    for gate in comparison.gates:
        name = quoteattr(gate.name)
        detail = escape(f"baseline={gate.baseline} candidate={gate.candidate} "
                        f"gate={gate.threshold}: {gate.message}")
        if gate.status == FAIL:
            body = f"<failure message={quoteattr(gate.message)}>{detail}</failure>"
        elif gate.status == WARN:
            body = f"<system-out>WARNING: {detail}</system-out>"
        else:
            body = f"<system-out>{detail}</system-out>"
        cases.append(f'    <testcase classname="golden_dataset_gate.{gate.kind}" name={name}>'
                     f"{body}</testcase>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuites name="golden-dataset-gate" tests="{len(comparison.gates)}" failures="{failures}">\n'
        f'  <testsuite name="golden-dataset-gate" tests="{len(comparison.gates)}" '
        f'failures="{failures}" errors="0" skipped="0">\n'
        + "\n".join(cases)
        + "\n  </testsuite>\n</testsuites>\n"
    )


def write_comparison(comparison: Comparison, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison.to_dict(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "comparison.md").write_text(render_markdown(comparison), encoding="utf-8")
    (output_dir / "junit.xml").write_text(render_junit(comparison), encoding="utf-8")
    with (output_dir / "changed-cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in comparison.changed_cases:
            handle.write(json.dumps(asdict(case)) + "\n")
