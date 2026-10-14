"""Command-line entry point: python -m app.cli <command>.

Exit codes: 0 success, 1 hard gate or check failed, 2 invalid input,
3 baseline update refused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.baseline import BaselineUpdateNotAllowed, approve_baseline
from app.change_scope import MOVING_GOALPOSTS_MESSAGE, changed_files, classify_changes
from app.comparator import compare, load_run, render_markdown, write_comparison
from app.evaluator import run_evaluation
from app.hashing import sha256_file
from app.schemas import DatasetError, category_counts, load_dataset, load_gate_policy

DEFAULT_CONFIG = "config/model.yaml"
DEFAULT_DATASET = "data/golden_dataset.jsonl"
DEFAULT_BASELINE = "baselines"
NULL_SHA = "0" * 40


def cmd_evaluate(args: argparse.Namespace) -> int:
    run = run_evaluation(args.config, args.dataset, args.output, model_override=args.model)
    s = run.summary
    print(f"Evaluated {s['total_cases']} cases with {run.manifest['provider']}/{run.manifest['model']}")
    print(f"  accuracy {s['category_accuracy']:.2%}  macro-F1 {s['macro_f1']:.2%}  "
          f"human-review {s['human_review_accuracy']:.2%}  critical safety errors "
          f"{s['critical_safety_errors']}  schema failures {s['schema_failures']}")
    print(f"  dataset sha256 {run.manifest['dataset_sha256']}")
    print(f"  wrote {args.output}/summary.json, predictions.jsonl, run-manifest.json")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    comparison = compare(load_run(args.baseline), load_run(args.candidate),
                         load_gate_policy(args.policy))
    write_comparison(comparison, args.output)
    print(render_markdown(comparison))
    print(f"Reports written to {args.output}/")
    return 0 if comparison.passed else 1


def cmd_approve_baseline(args: argparse.Namespace) -> int:
    try:
        if args.source:
            manifest = approve_baseline(args.source, args.baseline_dir, args.dataset)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                run_evaluation(args.config, args.dataset, tmp, model_override=args.model)
                manifest = approve_baseline(tmp, args.baseline_dir, args.dataset)
    except BaselineUpdateNotAllowed as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"Approved baseline {manifest['provider']}/{manifest['model']} "
          f"(dataset {manifest['dataset_sha256'][:12]}) into {args.baseline_dir}/")
    print(json.dumps(manifest["metrics"], indent=2))
    return 0


def cmd_validate_dataset(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)
    counts = category_counts(cases)
    print(f"{args.dataset}: {len(cases)} valid cases; {counts}")
    if args.expect_per_category is not None:
        wrong = {k: v for k, v in counts.items() if v != args.expect_per_category}
        if wrong:
            print(f"ERROR: expected {args.expect_per_category} cases per category, got {wrong}",
                  file=sys.stderr)
            return 1
    return 0


def cmd_verify_dataset_hash(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.baseline, "manifest.json").read_text(encoding="utf-8"))
    expected, actual = manifest["dataset_sha256"], sha256_file(args.dataset)
    if expected != actual:
        print("ERROR: the golden dataset does not match the dataset the approved baseline used.\n"
              f"  baseline: {expected}\n  current:  {actual}\n"
              "A dataset change needs its own reviewed PR followed by an explicit "
              "approve-baseline run.", file=sys.stderr)
        return 1
    print(f"Golden dataset hash verified: {actual}")
    return 0


def cmd_check_change_scope(args: argparse.Namespace) -> int:
    if args.files is not None:
        files = args.files
    elif not args.base or args.base == NULL_SHA:
        print("No base revision to compare against; change-scope check skipped.")
        return 0
    else:
        try:
            files = changed_files(args.base, args.head)
        except subprocess.CalledProcessError as exc:
            # Fail closed: an unverifiable change set is not assumed to be safe.
            print(f"ERROR: could not diff {args.base}...{args.head}: {exc.stderr.strip()}",
                  file=sys.stderr)
            return 2
    scope = classify_changes(files)
    if scope.violation:
        print(MOVING_GOALPOSTS_MESSAGE, file=sys.stderr)
        print(f"\nGolden dataset files: {', '.join(scope.dataset_files)}", file=sys.stderr)
        for area, paths in scope.system_files.items():
            print(f"{area}: {', '.join(paths)}", file=sys.stderr)
        return 1
    areas = ", ".join(scope.system_files) or "none"
    print(f"Change scope OK. Golden dataset changed: {bool(scope.dataset_files)}. "
          f"System areas changed: {areas}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evaluate", help="Run a configuration against a dataset")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default="reports/candidate")
    p.add_argument("--model", help="Override the model name from the config")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("gate", help="Compare a candidate with the baseline and apply gates")
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.add_argument("--candidate", default="reports/candidate")
    p.add_argument("--policy", default="config/gates.yaml")
    p.add_argument("--output", default="reports/comparison")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("approve-baseline", help="Replace the approved baseline (needs "
                       "ALLOW_BASELINE_UPDATE=true)")
    p.add_argument("--source", help="Existing run directory to promote; otherwise evaluate now")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--model", help="Override the model name from the config")
    p.add_argument("--baseline-dir", default=DEFAULT_BASELINE)
    p.set_defaults(func=cmd_approve_baseline)

    p = sub.add_parser("validate-dataset", help="Validate dataset schema and labels")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--expect-per-category", type=int)
    p.set_defaults(func=cmd_validate_dataset)

    p = sub.add_parser("verify-dataset-hash", help="Check the dataset matches the baseline")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--baseline", default=DEFAULT_BASELINE)
    p.set_defaults(func=cmd_verify_dataset_hash)

    p = sub.add_parser("check-change-scope", help="Fail if dataset and system changed together")
    p.add_argument("--base", help="Base git revision")
    p.add_argument("--head", default="HEAD")
    p.add_argument("--files", nargs="*", help="Explicit changed paths (skips git)")
    p.set_defaults(func=cmd_check_change_scope)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
