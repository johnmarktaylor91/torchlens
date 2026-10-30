"""Load benchmark tasks from their oracle directories.

A task directory (``oracles/<task-id>/``) holds the spec the agent reads, the inputs it may see,
and the private oracle material it must never see: the reference solution, the grader, held-out
inputs and seeded wrong solutions. An optional ``task.json`` manifest names each part; without
one the conventional names below apply. Directories whose names start with ``_`` or ``.`` (shared
material, caches) are not tasks.

Manifest fields (all optional)::

    {
      "id": "T12",                        # default: the directory name
      "title": "Control: one module's output",
      "spec": "spec.md",                  # the prompt shown to the agent
      "public": ["public"],               # files or folders copied into the agent's work folder
      "heldout": "heldout",               # private inputs the grader may use
      "grade": ["{python}", "grader.py", "--work", "{work}", "--heldout", "{heldout}",
                "--out", "{verdict}"],    # grader command, run from the task directory
      "arms": ["A", "B", "C", "D"],       # arms the task applies to
      "natives": {"B": ["pkg.module", "pkg.api:attribute"]},  # ready-made implementations to disable
      "grade_timeout_s": 900
    }

Grade placeholders: ``{python}`` the grader's interpreter, ``{arm_python}`` the arm's interpreter
(for running the agent's code), ``{work}`` the work folder or checkpoint snapshot, ``{heldout}``,
``{verdict}`` the JSON file the grader writes, ``{task_dir}``. The verdict holds ``passed`` and a
list of bars (``bars`` or ``checks``), each with ``name``, ``passed`` and optionally ``value`` and
``threshold``. A grader that exits nonzero, writes no verdict or omits ``passed`` fails the run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Default oracle root, beside this module.
ORACLE_ROOT = Path(__file__).resolve().parent / "oracles"
#: Every arm the benchmark knows.
ALL_ARMS: tuple[str, ...] = ("A", "B", "C", "D")
DEFAULT_GRADE: tuple[str, ...] = (
    "{python}",
    "grader.py",
    "--work",
    "{work}",
    "--heldout",
    "{heldout}",
    "--out",
    "{verdict}",
)
#: Names that are private by convention even when a manifest does not list them.
PRIVATE_NAMES = frozenset(
    {"reference", "reference.py", "grader.py", "heldout", "wrong", "seeded", "tests", "task.json"}
)


@dataclass(frozen=True)
class BenchTask:
    """One task loaded from its oracle directory."""

    task_id: str
    title: str
    root: Path
    spec: Path
    public: tuple[Path, ...]
    heldout: Path | None
    grade: tuple[str, ...]
    arms: tuple[str, ...]
    natives: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    grade_timeout_s: float = 900.0

    def prompt(self) -> str:
        """Return the spec text the agent reads."""

        return self.spec.read_text(encoding="utf-8")

    def applies_to(self, arm: str) -> bool:
        """Return whether the task is run in ``arm`` (others are reported not applicable)."""

        return arm in self.arms

    def grade_argv(self, values: Mapping[str, str]) -> list[str]:
        """Fill the grade command's placeholders from ``values``."""

        return [part.format(**values) for part in self.grade]


def _as_paths(root: Path, names: Sequence[str]) -> tuple[Path, ...]:
    paths = []
    for name in names:
        path = (root / name).resolve()
        if root.resolve() not in path.parents and path != root.resolve():
            raise ValueError(f"{root.name}: public path {name!r} leaves the task directory")
        if path.name in PRIVATE_NAMES:
            raise ValueError(f"{root.name}: {name!r} is oracle material and cannot be public")
        if path.exists():
            paths.append(path)
    return tuple(paths)


def _natives(raw: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    unknown = sorted(set(raw) - set(ALL_ARMS))
    if unknown:
        raise ValueError(f"natives names unknown arms {unknown}")
    return {arm: tuple(str(item) for item in items) for arm, items in raw.items()}


def load_task(root: Path) -> BenchTask:
    """Load one task directory.

    Parameters
    ----------
    root:
        The task directory; it must hold the spec (``spec.md`` unless the manifest names another).

    Returns
    -------
    BenchTask
        The task, with every path resolved inside ``root``.
    """

    manifest_path = root / "task.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = root / manifest.get("spec", "spec.md")
    if not spec.is_file():
        raise FileNotFoundError(f"{root.name}: no spec at {spec.name}")
    heldout_name = manifest.get("heldout", "heldout")
    heldout = root / heldout_name if (root / heldout_name).exists() else None
    arms = tuple(manifest.get("arms", ALL_ARMS))
    if not set(arms) <= set(ALL_ARMS):
        raise ValueError(f"{root.name}: unknown arms in {arms}")
    return BenchTask(
        task_id=str(manifest.get("id", root.name)),
        title=str(manifest.get("title", root.name)),
        root=root.resolve(),
        spec=spec.resolve(),
        public=_as_paths(root, manifest.get("public", ["public"])),
        heldout=heldout.resolve() if heldout else None,
        grade=tuple(manifest.get("grade", DEFAULT_GRADE)),
        arms=arms,
        natives=_natives(manifest.get("natives", {})),
        grade_timeout_s=float(manifest.get("grade_timeout_s", 900.0)),
    )


def load_registry(root: Path = ORACLE_ROOT) -> dict[str, BenchTask]:
    """Load every task directory under ``root``, keyed by task id.

    Raises
    ------
    ValueError
        If two directories declare the same task id.
    """

    tasks: dict[str, BenchTask] = {}
    if not root.is_dir():
        return tasks
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        task = load_task(child)
        if task.task_id in tasks:
            raise ValueError(f"duplicate task id {task.task_id} in {child.name}")
        tasks[task.task_id] = task
    return tasks


def normalize_verdict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a grader verdict in the record's shape: ``passed`` and ``bars``.

    A verdict without a boolean ``passed`` is a failure; ``passed`` is also false when any bar
    failed, so a grader cannot report a pass over a failing bar.
    """

    bars_raw = raw.get("bars", raw.get("checks", []))
    bars = [
        {
            "name": str(bar.get("name", f"bar{index}")),
            "passed": bar.get("passed") is True,
            "value": bar.get("value"),
            "threshold": bar.get("threshold"),
        }
        for index, bar in enumerate(bars_raw)
        if isinstance(bar, Mapping)
    ]
    declared = raw.get("passed")
    passed = declared is True and all(bar["passed"] for bar in bars)
    verdict: dict[str, Any] = {"passed": passed, "bars": bars}
    if declared is not True and declared is not False:
        verdict["error"] = "verdict has no boolean 'passed'"
    return verdict
