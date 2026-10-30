"""Load benchmark tasks from their oracle directories.

A task directory (``oracles/<task-id>/``) follows the oracle authors' solution contract
(``oracles/_common/contract.md``): ``spec.md`` (its "Prompt to the agent" section is what the
agent reads), ``inputs/`` (the inputs the agent may see), and private material the agent never
sees: ``reference/solve.py``, ``grader.py``, ``heldout/`` (inputs in the same layout, for a second
run) and seeded wrong solutions under ``wrong/<name>/solve.py``. A solution is a folder holding
``solve.py``, run as ``python solve.py --inputs DIR --outputs DIR`` from that folder. The grader
runs as ``python grader.py --task-dir TASK --outputs OUT [--heldout-outputs HOUT] [--cache DIR]``
and prints one JSON verdict on its last stdout line. Directories whose names start with ``_`` or
``.`` (shared material, caches) are not tasks.

``task.json`` fields (all optional)::

    {
      "id": "t12-control-module-output",   # default: the directory name
      "title": "...",
      "timeout_s": 600,                     # limit for one solve.py run
      "heldout": true,                      # run solve.py again on heldout/ and grade both
      "arms": ["A", "B", "C", "D"],         # arms the task applies to
      "natives": {"B": ["pkg.module"]},     # extra ready-made implementations to disable
      "grade_timeout_s": 1800
    }

The verdict holds ``passed`` and a list of bars, each with ``name``, ``passed`` and optionally
``value`` and ``threshold``. A verdict that is missing, unparsable or has no boolean ``passed``
fails the run, as does any failing bar. The grader's exit code is recorded, not trusted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Default oracle root, beside this module.
ORACLE_ROOT = Path(__file__).resolve().parent / "oracles"
#: Every arm the benchmark knows.
ALL_ARMS: tuple[str, ...] = ("A", "B", "C", "D")
PROMPT_HEADING = "## Prompt to the agent"


@dataclass(frozen=True)
class BenchTask:
    """One task loaded from its oracle directory."""

    task_id: str
    title: str
    root: Path
    spec: Path
    inputs: Path | None
    heldout: Path | None
    arms: tuple[str, ...]
    natives: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    solve_timeout_s: float = 1800.0
    grade_timeout_s: float = 1800.0

    def prompt(self) -> str:
        """Return what the agent reads: the spec's prompt section (the whole spec if it has none).

        The rest of the spec (oracle method, pass bars, held-out design) stays with the grader.
        """

        return prompt_section(self.spec.read_text(encoding="utf-8"))

    def applies_to(self, arm: str) -> bool:
        """Return whether the task is run in ``arm`` (others are reported not applicable)."""

        return arm in self.arms

    def grade_argv(
        self, python: str, outputs: Path, heldout_outputs: Path | None, cache: Path
    ) -> list[str]:
        """Return the grader command for one graded solution."""

        argv = [python, str(self.root / "grader.py"), "--task-dir", str(self.root)]
        argv += ["--outputs", str(outputs), "--cache", str(cache)]
        if heldout_outputs is not None:
            argv += ["--heldout-outputs", str(heldout_outputs)]
        return argv


def prompt_section(spec: str) -> str:
    """Return the text under ``## Prompt to the agent`` up to the next level-2 heading."""

    lines = spec.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == PROMPT_HEADING)
    except StopIteration:
        return spec
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    title = next((line for line in lines if line.startswith("# ")), "")
    return "\n".join([title, "", *body]).strip() + "\n"


def _natives(raw: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    unknown = sorted(set(raw) - {"*", *ALL_ARMS})
    if unknown:
        raise ValueError(f"natives names unknown arms {unknown}")
    return {arm: tuple(str(item) for item in items) for arm, items in raw.items()}


def load_task(root: Path) -> BenchTask:
    """Load one task directory.

    Parameters
    ----------
    root:
        The task directory; it must hold ``spec.md`` and ``grader.py``.

    Returns
    -------
    BenchTask
        The task, with every path resolved inside ``root``.
    """

    manifest_path = root / "task.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for required in ("spec.md", "grader.py"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"{root.name}: no {required}")
    arms = tuple(manifest.get("arms", ALL_ARMS))
    if not set(arms) <= set(ALL_ARMS):
        raise ValueError(f"{root.name}: unknown arms in {arms}")
    heldout = root / "heldout"
    use_heldout = bool(manifest.get("heldout", heldout.is_dir()))
    if use_heldout and not heldout.is_dir():
        raise FileNotFoundError(f"{root.name}: task.json asks for heldout/ but it is missing")
    inputs = root / "inputs"
    return BenchTask(
        task_id=str(manifest.get("id", root.name)),
        title=str(manifest.get("title", root.name)),
        root=root.resolve(),
        spec=(root / "spec.md").resolve(),
        inputs=inputs.resolve() if inputs.is_dir() else None,
        heldout=heldout.resolve() if use_heldout else None,
        arms=arms,
        natives=_natives(manifest.get("natives", {})),
        solve_timeout_s=float(manifest.get("timeout_s", 1800.0)),
        grade_timeout_s=float(manifest.get("grade_timeout_s", 1800.0)),
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


def parse_verdict(stdout: str) -> dict[str, Any]:
    """Return the normalized verdict from a grader's stdout (its last non-empty line)."""

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {"passed": False, "bars": [], "error": "grader printed nothing"}
    try:
        raw = json.loads(lines[-1])
    except ValueError:
        return {"passed": False, "bars": [], "error": "grader's last line is not JSON"}
    if not isinstance(raw, Mapping):
        return {"passed": False, "bars": [], "error": "grader's verdict is not an object"}
    return normalize_verdict(raw)


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
