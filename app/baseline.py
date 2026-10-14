"""Explicit, human-approved baseline management.

Passing the gate never redefines "correct". A new baseline is written only by
`approve-baseline` with ALLOW_BASELINE_UPDATE=true set by a person.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.evaluator import PREDICTIONS_FILE, RUN_MANIFEST_FILE, SUMMARY_FILE, git_commit
from app.hashing import sha256_file

BASELINE_MANIFEST_FILE = "manifest.json"
ALLOW_FLAG = "ALLOW_BASELINE_UPDATE"


class BaselineUpdateNotAllowed(PermissionError):
    pass


def baseline_update_allowed(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get(ALLOW_FLAG, "").strip().lower() == "true"


def approve_baseline(source_dir: str | Path, baseline_dir: str | Path, dataset_path: str | Path,
                     env: dict[str, str] | None = None) -> dict:
    """Promote a completed run to be the approved baseline."""
    if not baseline_update_allowed(env):
        raise BaselineUpdateNotAllowed(
            f"Refusing to replace the approved baseline. Set {ALLOW_FLAG}=true to approve "
            "a new baseline deliberately. CI never sets this flag."
        )
    source_dir, baseline_dir = Path(source_dir), Path(baseline_dir)
    for name in (SUMMARY_FILE, PREDICTIONS_FILE, RUN_MANIFEST_FILE):
        if not (source_dir / name).exists():
            raise FileNotFoundError(f"{source_dir / name} is missing; run `evaluate` first")

    manifest = json.loads((source_dir / RUN_MANIFEST_FILE).read_text(encoding="utf-8"))
    current_hash = sha256_file(dataset_path)
    if manifest["dataset_sha256"] != current_hash:
        raise ValueError(
            "The run was produced from a different golden dataset than the current one "
            f"({manifest['dataset_sha256'][:12]} vs {current_hash[:12]}). Re-run evaluate."
        )

    env = os.environ if env is None else env
    manifest["approval"] = {
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved_by": env.get("GITHUB_ACTOR") or env.get("USER") or _safe_user(),
        "approved_at_commit": git_commit(),
        "source_run": source_dir.as_posix(),
    }

    baseline_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / SUMMARY_FILE, baseline_dir / SUMMARY_FILE)
    shutil.copyfile(source_dir / PREDICTIONS_FILE, baseline_dir / PREDICTIONS_FILE)
    (baseline_dir / BASELINE_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - platform dependent
        return "unknown"
