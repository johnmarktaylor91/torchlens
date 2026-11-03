"""Prepare, run and grade agent-from-docs tasks.

Usage (from the repository root)::

    python tools/agent_evals/harness.py offline [--tasks ID,...] [--out DIR]
    python tools/agent_evals/harness.py prepare --out DIR [--tasks ID,...]
    python tools/agent_evals/harness.py grade --dir DIR [--tasks ID,...]
    python tools/agent_evals/harness.py live --agent-cmd TEMPLATE --out DIR [--tasks ID,...]

``offline`` copies each reference solution into a fresh task directory, executes it and grades
it (the test suite runs this path). ``prepare`` writes task directories (prompt, fixture model,
docs bundle) for an agent to fill in; ``grade`` executes and grades whatever ``solution.py``
files exist. ``live`` runs ``prepare``, then the agent command once per task, then ``grade``.
The agent command template may use ``{prompt}`` (prompt file), ``{docs}`` (docs directory),
``{workdir}`` (task directory) and ``{solution}`` (the file the agent must write); it runs with
the task directory as its working directory. Live mode calls a model and is never run in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_evals.tasks import MODEL_SOURCE, TASKS, TASKS_BY_ID, Task  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path(__file__).resolve().parent / "references"
#: Wall-clock limit for one solution script.
SCRIPT_TIMEOUT_S = 120
#: Wall-clock limit for one live agent invocation.
AGENT_TIMEOUT_S = 900


def build_docs_bundle(dest: Path) -> Path:
    """Copy the shipped documentation (Markdown only, no source) into ``dest``.

    The bundle is ``README.md``, every Markdown page under ``docs/`` and the agent guide that
    ``torchlens.agent.guide()`` serves (``agent_guide.md``).

    Parameters
    ----------
    dest:
        Directory to create; it must not exist yet.

    Returns
    -------
    Path
        ``dest``.
    """

    from torchlens.agent import guide

    dest.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "README.md", dest / "README.md")
    for page in sorted((REPO_ROOT / "docs").rglob("*.md")):
        target = dest / "docs" / page.relative_to(REPO_ROOT / "docs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(page, target)
    (dest / "agent_guide.md").write_text(guide(), encoding="utf-8")
    return dest


def prepare_task(task: Task, task_dir: Path, docs_dir: Path) -> Path:
    """Write one task directory: ``prompt.md`` and the fixture ``model.py``.

    The prompt names the docs directory relative to the task directory.
    """

    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "model.py").write_text(MODEL_SOURCE, encoding="utf-8")
    # Relative, so a prepared tree can move to the machine the agent runs on.
    docs_ref = os.path.relpath(docs_dir, task_dir)
    (task_dir / "prompt.md").write_text(task.full_prompt(docs_ref), encoding="utf-8")
    return task_dir


def run_solution(task_dir: Path, *, timeout: float = SCRIPT_TIMEOUT_S) -> dict[str, Any]:
    """Execute ``solution.py`` in ``task_dir`` and return its exit facts."""

    script = task_dir / "solution.py"
    if not script.exists():
        return {"ran": False, "returncode": None, "stderr_tail": "solution.py missing"}
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OMP_NUM_THREADS", "1")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, script.name],
            cwd=task_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ran": True, "returncode": None, "stderr_tail": f"timed out after {timeout}s"}
    return {
        "ran": True,
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - start, 2),
        "stderr_tail": proc.stderr[-2000:],
    }


def grade_task(task: Task, task_dir: Path) -> dict[str, Any]:
    """Execute and grade one task directory.

    Returns
    -------
    dict[str, Any]
        ``task``, ``passed``, ``run`` (exit facts) and ``checks`` (name, passed, detail).
    """

    run = run_solution(task_dir)
    checks: list[tuple[str, bool, str]] = [
        ("script exits 0", run["returncode"] == 0, run["stderr_tail"] if run["ran"] else "")
    ]
    if run["returncode"] == 0:
        try:
            checks.extend(task.grade(task_dir))
        except Exception as exc:  # noqa: BLE001 - a grader crash is a failed check, never a pass
            checks.append(("grader ran", False, f"{type(exc).__name__}: {exc}"))
    return {
        "task": task.task_id,
        "passed": all(passed for _, passed, _ in checks),
        "run": run,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
    }


def select_tasks(ids: str | None) -> list[Task]:
    """Resolve a comma-separated id list (``None`` means every task)."""

    if not ids:
        return list(TASKS)
    unknown = [item for item in ids.split(",") if item not in TASKS_BY_ID]
    if unknown:
        raise SystemExit(f"unknown task ids {unknown}; known: {sorted(TASKS_BY_ID)}")
    return [TASKS_BY_ID[item] for item in ids.split(",")]


def run_offline(tasks: list[Task], out: Path) -> list[dict[str, Any]]:
    """Run every task's reference solution in a fresh directory under ``out``."""

    results = []
    for task in tasks:
        task_dir = prepare_task(task, out / task.task_id, Path("docs-not-needed-offline"))
        shutil.copy2(REFERENCE_DIR / f"{task.task_id}.py", task_dir / "solution.py")
        results.append(grade_task(task, task_dir))
    return results


def run_live(tasks: list[Task], out: Path, agent_cmd: str) -> list[dict[str, Any]]:
    """Prepare, invoke the agent command per task, then grade."""

    docs_dir = build_docs_bundle(out / "docs")
    results = []
    for task in tasks:
        task_dir = prepare_task(task, out / task.task_id, docs_dir)
        command = agent_cmd.format(
            prompt=shlex.quote(str(task_dir / "prompt.md")),
            docs=shlex.quote(str(docs_dir)),
            workdir=shlex.quote(str(task_dir)),
            solution=shlex.quote(str(task_dir / "solution.py")),
        )
        try:
            agent = subprocess.run(
                command, shell=True, cwd=task_dir, timeout=AGENT_TIMEOUT_S, check=False
            )
            agent_rc: int | None = agent.returncode
        except subprocess.TimeoutExpired:
            agent_rc = None
        result = grade_task(task, task_dir)
        result["agent_returncode"] = agent_rc
        results.append(result)
    return results


def _summarize(results: list[dict[str, Any]]) -> str:
    lines = []
    for result in results:
        failed = [check for check in result["checks"] if not check["passed"]]
        reason = "; ".join(f"{c['name']}: {c['detail'][:160]}" for c in failed)
        lines.append(
            f"{result['task']}: {'PASS' if result['passed'] else 'FAIL'} {reason}".rstrip()
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; exit 0 when every graded task passed."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("offline", "prepare", "grade", "live"):
        sp = sub.add_parser(mode)
        sp.add_argument("--tasks", default=None, help="comma-separated task ids (default: all)")
        if mode == "grade":
            sp.add_argument("--dir", type=Path, required=True)
        else:
            sp.add_argument("--out", type=Path, required=mode != "offline")
        if mode == "live":
            sp.add_argument("--agent-cmd", required=True)
    args = parser.parse_args(argv)
    tasks = select_tasks(args.tasks)
    if args.mode == "prepare":
        docs_dir = build_docs_bundle(args.out / "docs")
        for task in tasks:
            prepare_task(task, args.out / task.task_id, docs_dir)
        print(f"prepared {len(tasks)} tasks under {args.out}")
        return 0
    if args.mode == "offline":
        out = args.out or Path(os.environ.get("TMPDIR", "/tmp")) / "torchlens-agent-evals"
        shutil.rmtree(out, ignore_errors=True)
        results = run_offline(tasks, out)
    elif args.mode == "grade":
        results = [grade_task(task, args.dir / task.task_id) for task in tasks]
    else:
        results = run_live(tasks, args.out, args.agent_cmd)
    report_dir = args.dir if args.mode == "grade" else (args.out or out)
    (report_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(_summarize(results))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
