"""Detect a golden-dataset change made in the same change set as a system change.

Changing the expected answers and the system together lets a weak candidate pass by
moving the goalposts. The dataset must change in its own reviewed pull request.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatch

GOLDEN_DATASET_PATTERNS = ("data/golden_dataset.jsonl",)

SYSTEM_BEHAVIOR_PATTERNS: dict[str, tuple[str, ...]] = {
    "Model configuration": ("config/model*.yaml",),
    "Prompt": ("config/prompt*.txt",),
    "Provider implementation": ("app/providers/*",),
    "Evaluation logic": ("app/evaluator.py", "app/metrics.py", "app/safety.py",
                         "app/comparator.py", "app/schemas.py"),
    "Gate thresholds": ("config/gates.yaml",),
}

MOVING_GOALPOSTS_MESSAGE = (
    "The candidate system and its golden dataset changed together.\n\n"
    "Evaluate system changes against the existing golden dataset first.\n"
    "Update the golden dataset through a separate reviewed change."
)


@dataclass
class ChangeScope:
    dataset_files: list[str] = field(default_factory=list)
    system_files: dict[str, list[str]] = field(default_factory=dict)

    @property
    def violation(self) -> bool:
        return bool(self.dataset_files) and bool(self.system_files)


def classify_changes(paths: list[str]) -> ChangeScope:
    scope = ChangeScope()
    for raw in paths:
        path = raw.strip().replace("\\", "/")
        if not path:
            continue
        if any(fnmatch(path, p) for p in GOLDEN_DATASET_PATTERNS):
            scope.dataset_files.append(path)
            continue
        for area, patterns in SYSTEM_BEHAVIOR_PATTERNS.items():
            if any(fnmatch(path, p) for p in patterns):
                scope.system_files.setdefault(area, []).append(path)
                break
    return scope


def changed_files(base: str, head: str = "HEAD") -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]
