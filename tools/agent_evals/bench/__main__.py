"""Command line for the agent benchmark (run from ``tools/``: ``python -m agent_evals.bench``).

Commands::

    list         [--oracles DIR]
    build-env    --arm A --dest DIR [--repo DIR] [--oracles DIR]
    check-task   --task ID --python PY [--oracles DIR] [--out DIR]
    plan         --out FILE --seed N [--pilot | --tasks T1,T7 --arms A,B --tiers small,mid]
                 [--repeats 3]
    run          --task ID --arm A --tier small --config FILE --envs DIR --out DIR
                 [--repeat 0] [--oracles DIR] [--material-root DIR]
    run-plan     --plan FILE --config FILE --envs DIR --out DIR [--limit N] [--oracles DIR]
                 [--material-root DIR]
    summarize    --results DIR

``build-env`` builds one arm's environment, audits its imports, installs the guard and checks
that every guard entry fires; it writes ``DIR/env.json``. ``check-task`` runs a task's reference
and seeded wrong solutions through its grader. ``run`` and ``run-plan`` call subject agents and
are never run in CI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import arms, guards, registry, runner, scheduler, telemetry

REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIAL_ROOT = registry.ORACLE_ROOT / "_material"


def _tasks(args: argparse.Namespace) -> dict[str, registry.BenchTask]:
    return registry.load_registry(Path(args.oracles))


def cmd_list(args: argparse.Namespace) -> int:
    for task in _tasks(args).values():
        print(f"{task.task_id}\t{','.join(task.arms)}\t{task.title}")
    return 0


def cmd_build_env(args: argparse.Namespace) -> int:
    arm = arms.ARMS[args.arm]
    dest = Path(args.dest)
    python = arms.build(arm, dest, Path(args.repo))
    report: dict[str, Any] = {"audit": arms.audit(arm, python), "guards": {}}
    clean_env = {"PATH": f"{python.parent}:/usr/bin:/bin", "HOME": str(dest)}
    if arm.library:
        # With no run file the guard applies every task's entries; the library must still load.
        probe = subprocess.run(
            [str(python), "-c", f"import {arm.library}"],
            env=clean_env,
            capture_output=True,
            text=True,
        )
        report["library_imports_under_union"] = probe.returncode == 0
        report["union_import_error"] = probe.stderr[-1000:]
    for task_id in guards.load_natives():
        entries = guards.entries_for(task_id, arm.arm_id)
        if not entries:
            continue
        config = guards.write_run_config(dest / f"guard-{task_id}.json", entries)
        env = {**clean_env, guards.ENV_VAR: str(config)}
        results = guards.selftest(str(python), entries, env)
        imports_ok = True
        if arm.library:
            probe = subprocess.run(
                [str(python), "-c", f"import {arm.library}"],
                env=env,
                capture_output=True,
                text=True,
            )
            imports_ok = probe.returncode == 0
        report["guards"][task_id] = {
            "results": results,
            "library_still_imports": imports_ok,
            "ok": imports_ok and all(r["outcome"] != "not blocked" for r in results),
        }
    report["ok"] = (
        report["audit"]["ok"]
        and report.get("library_imports_under_union", True)
        and all(g["ok"] for g in report["guards"].values())
    )
    (dest / "env.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"arm": arm.arm_id, "ok": report["ok"], "leaked": report["audit"]["leaked"]}))
    return 0 if report["ok"] else 1


def _grade_candidate(
    task: registry.BenchTask, candidate: Path, python: str, out: Path
) -> dict[str, Any]:
    work = out / "work"
    work.mkdir(parents=True)
    for path in task.public:
        (shutil.copytree if path.is_dir() else shutil.copy2)(path, work / path.name)
    shutil.copy2(candidate, work / "solution.py")
    subprocess.run(
        [python, "solution.py"], cwd=work, capture_output=True, timeout=1800, check=False
    )
    spec = runner.RunSpec(task, "A", "small", 0, Path(python), Path(python))
    return runner.grade(spec, work, out / "grade")


def cmd_check_task(args: argparse.Namespace) -> int:
    task = _tasks(args)[args.task]
    out = Path(args.out or tempfile.mkdtemp(prefix=f"check-{task.task_id}-"))
    results = {
        "reference": _grade_candidate(
            task, task.root / "reference.py", args.python, out / "reference"
        )
    }
    for wrong in sorted((task.root / "wrong").glob("*.py")):
        results[f"wrong/{wrong.name}"] = _grade_candidate(
            task, wrong, args.python, out / wrong.stem
        )
    ok = results["reference"]["passed"] and not any(
        v["passed"] for k, v in results.items() if k != "reference"
    )
    print(json.dumps({k: v["passed"] for k, v in results.items()} | {"ok": ok}))
    return 0 if ok else 1


def cmd_plan(args: argparse.Namespace) -> int:
    if args.pilot:
        cells = scheduler.pilot_cells()
    else:
        cells = [
            (t, a, r)
            for t in args.tasks.split(",")
            for a in args.arms.split(",")
            for r in args.tiers.split(",")
        ]
    rows = scheduler.plan(cells, args.repeats, args.seed)
    scheduler.write(rows, Path(args.out), args.seed)
    print(json.dumps({"rows": len(rows), "out": args.out}))
    return 0


def _spec(
    args: argparse.Namespace, task: registry.BenchTask, arm_id: str, tier: str, repeat: int
) -> runner.RunSpec:
    arm = arms.ARMS[arm_id]
    envs = Path(args.envs)
    env_report = envs / arm_id / "env.json"
    environment = {"arm_env": str(envs / arm_id)}
    if env_report.exists():
        audit = json.loads(env_report.read_text())["audit"]
        environment["freeze_sha256"] = audit["freeze_sha256"]
    material = Path(args.material_root) / arm.material if arm.material else None
    return runner.RunSpec(
        task=task,
        arm=arm_id,
        tier=tier,
        repeat=repeat,
        arm_python=envs / arm_id / "bin" / "python",
        grader_python=envs / "A" / "bin" / "python",
        material=material if material and material.exists() else None,
        library=arm.library,
        environment=environment,
    )


def cmd_run(args: argparse.Namespace) -> int:
    config = runner.RunConfig.load(Path(args.config))
    task = _tasks(args)[args.task]
    spec = _spec(args, task, args.arm, args.tier, args.repeat)
    record = runner.run_one(spec, Path(args.out), config)
    print(
        json.dumps(
            {k: record[k] for k in ("run_id", "task", "arm", "tier", "end_reason", "passed")}
        )
    )
    return 0


def cmd_run_plan(args: argparse.Namespace) -> int:
    config = runner.RunConfig.load(Path(args.config))
    tasks = _tasks(args)
    rows = [json.loads(line) for line in Path(args.plan).read_text().splitlines()[1:]]
    out = Path(args.out)
    done = 0
    for row in rows:
        if args.limit is not None and done >= args.limit:
            break
        run_dir = out / row["run_key"]
        if (run_dir / "record.json").exists():
            continue
        spec = _spec(args, tasks[row["task"]], row["arm"], row["tier"], row["repeat"])
        record = runner.run_one(spec, run_dir, config)
        print(
            json.dumps(
                {
                    "run_key": row["run_key"],
                    "end_reason": record["end_reason"],
                    "passed": record["passed"],
                }
            )
        )
        done += 1
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    records = [json.loads(p.read_text()) for p in sorted(Path(args.results).glob("*/record.json"))]
    problems = {r.get("run_id", "?"): telemetry.validate_record(r) for r in records}
    cells: dict[str, dict[str, int]] = {}
    for record in records:
        key = f"{record['task']}/{record['arm']}/{record['tier']}"
        cell = cells.setdefault(key, {"runs": 0, "passed": 0})
        cell["runs"] += 1
        cell["passed"] += int(record["passed"])
    print(
        json.dumps(
            {
                "runs": len(records),
                "invalid": {k: v for k, v in problems.items() if v},
                "cells": cells,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_evals.bench")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Any) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--oracles", default=str(registry.ORACLE_ROOT))
        return p

    add("list", cmd_list)
    p = add("build-env", cmd_build_env)
    p.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    p.add_argument("--dest", required=True)
    p.add_argument("--repo", default=str(REPO_ROOT))
    p = add("check-task", cmd_check_task)
    p.add_argument("--task", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--out")
    p = add("plan", cmd_plan)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--tasks", default="T12")
    p.add_argument("--arms", default="A")
    p.add_argument("--tiers", default="small")
    p.add_argument("--repeats", type=int, default=3)
    for name, func in (("run", cmd_run), ("run-plan", cmd_run_plan)):
        p = add(name, func)
        p.add_argument("--config", required=True)
        p.add_argument("--envs", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--material-root", default=str(MATERIAL_ROOT))
        if name == "run":
            p.add_argument("--task", required=True)
            p.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
            p.add_argument("--tier", required=True, choices=runner.TIERS)
            p.add_argument("--repeat", type=int, default=0)
        else:
            p.add_argument("--plan", required=True)
            p.add_argument("--limit", type=int)
    p = add("summarize", cmd_summarize)
    p.add_argument("--results", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
