"""Command line for the agent benchmark (run from ``tools/``: ``python -m agent_evals.bench``).

Commands::

    list         [--oracles DIR]
    build-env    --arm A --dest DIR [--repo DIR] [--oracles DIR]
    prefetch     --cache DIR --python PY [--tasks T1,T7]
    check-task   --task ID --python PY [--oracles DIR] [--out DIR] [--model-cache DIR]
    plan         --out FILE --seed N [--pilot | --tasks T1,T7 --arms A,B --tiers small,mid]
                 [--repeats 3]
    run          --task ID --arm A --tier small --config FILE --envs DIR --out DIR
                 [--repeat 0] [--oracles DIR] [--material-root DIR]
    stage        --task ID --arm A --tier small --envs DIR --dest DIR    (external runner)
    ingest       --task ID --arm A --tier small --envs DIR --returned DIR --out DIR
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
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import arms, external, guards, registry, runner, scheduler, telemetry

REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIAL_ROOT = registry.ORACLE_ROOT / "_material"


class _Tasks(dict):  # type: ignore[type-arg]
    """Tasks by id that also resolve the plan's short names (``T1`` for ``t01-jacobian-lens``)."""

    def __missing__(self, key: str) -> registry.BenchTask:
        matches = [t for i, t in self.items() if guards.task_number(i) == guards.task_number(key)]
        if len(matches) != 1:
            raise SystemExit(f"no single task matches {key!r}; known: {sorted(self)}")
        return matches[0]


def _tasks(args: argparse.Namespace) -> dict[str, registry.BenchTask]:
    return _Tasks(registry.load_registry(Path(args.oracles)))


def cmd_list(args: argparse.Namespace) -> int:
    for task in _tasks(args).values():
        print(f"{task.task_id}\t{','.join(task.arms)}\t{task.title}")
    return 0


def cmd_build_env(args: argparse.Namespace) -> int:
    arm = arms.ARMS[args.arm]
    dest = Path(args.dest)
    natives = guards.load_natives()
    if args.tasks:
        # A per-task environment: its fail-closed guard holds only these tasks' entries, for a
        # runner that clears the environment (so no per-run guard file reaches the agent).
        wanted = {guards.task_number(t) for t in args.tasks.split(",")}
        natives = {t: v for t, v in natives.items() if t in wanted}
    python = arms.build(arm, dest, Path(args.repo), natives=natives)
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
    report["tasks"] = sorted(natives)
    report["union"] = guards.union_for(arm.arm_id, natives)
    for task_id in natives:
        entries = guards.entries_for(task_id, arm.arm_id, natives=natives)
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
    task: registry.BenchTask, candidate: Path, python: str, out: Path, config: runner.RunConfig
) -> dict[str, Any]:
    # Run in place: seeded wrong solutions import the reference by its path in the task tree.
    solution = candidate
    out.mkdir(parents=True)
    spec = runner.RunSpec(task, "A", "small", 0, Path(python), Path(python))
    guards.write_run_config(out / "guard.json", [])
    env = runner.agent_env(spec, config, out)
    return runner.grade(spec, solution, out / "grade", env, config)


def cmd_check_task(args: argparse.Namespace) -> int:
    """Grade the task's reference, every seeded wrong solution and the do-nothing solution.

    This checks the harness's wiring of the oracle authors' graders: the reference must pass and
    every other candidate must fail.
    """

    task = _tasks(args)[args.task]
    out = Path(args.out or tempfile.mkdtemp(prefix=f"check-{task.task_id}-"))
    config = runner.RunConfig(
        agent_cmd={},
        caps={},
        model_cache=Path(args.model_cache) if args.model_cache else None,
        oracle_cache=out / "oracle_cache",
    )
    candidates = {"reference": task.root / "reference"}
    candidates |= {
        f"wrong/{d.name}": d for d in sorted((task.root / "wrong").glob("*")) if d.is_dir()
    }
    do_nothing = Path(args.oracles) / "_common" / "do_nothing"
    if do_nothing.is_dir():
        candidates["do_nothing"] = do_nothing
    results = {
        name: _grade_candidate(task, path, args.python, out / name.replace("/", "-"), config)
        for name, path in candidates.items()
    }
    ok = results["reference"]["passed"] and not any(
        v["passed"] for k, v in results.items() if k != "reference"
    )
    summary = {k: v["passed"] for k, v in results.items()} | {"ok": ok}
    (out / "check.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(summary))
    return 0 if ok else 1


MODELS_FILE = Path(__file__).resolve().parent / "models.json"
PREFETCH = """
import json, os, sys
item = json.loads(sys.argv[1])
if item["kind"] == "hf":
    from huggingface_hub import snapshot_download
    print(snapshot_download(item["id"]))
elif item["kind"] == "torchvision":
    import torchvision
    holder, name = item["weights"].split(".")
    weights = getattr(getattr(torchvision.models, holder), name)
    getattr(torchvision.models, item["builder"])(weights=weights)
    print(os.environ["TORCH_HOME"])
else:
    raise SystemExit("unknown model kind " + item["kind"])
"""


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Download every task's pretrained weights into the model cache (needs network)."""

    cache = Path(args.cache).resolve()
    env = {**os.environ, **runner.cache_env(cache)}
    table = json.loads(MODELS_FILE.read_text())["tasks"]
    wanted = args.tasks.split(",") if args.tasks else sorted(table)
    status = 0
    for task in wanted:
        for item in table.get(guards.task_number(task), []):
            proc = subprocess.run(
                [args.python, "-c", PREFETCH, json.dumps(item)],
                env=env,
                capture_output=True,
                text=True,
            )
            print(json.dumps({"task": task, "model": item, "ok": proc.returncode == 0}))
            if proc.returncode != 0:
                print(proc.stderr[-1500:], file=sys.stderr)
                status = 1
    return status


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
    # A per-task environment (``B-T7``) wins over the arm's shared one (``B``).
    env_dir = envs / f"{arm_id}-{guards.task_number(task.task_id)}"
    env_dir = env_dir if env_dir.is_dir() else envs / arm_id
    env_report = env_dir / "env.json"
    environment = {"arm_env": str(env_dir)}
    if env_report.exists():
        audit = json.loads(env_report.read_text())["audit"]
        environment["freeze_sha256"] = audit["freeze_sha256"]
    material = Path(args.material_root) / arm.material if arm.material else None
    return runner.RunSpec(
        task=task,
        arm=arm_id,
        tier=tier,
        repeat=repeat,
        arm_python=env_dir / "bin" / "python",
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


def cmd_stage(args: argparse.Namespace) -> int:
    """Write one run's task folder and prompt for an external subject runner."""

    task = _tasks(args)[args.task]
    spec = _spec(args, task, args.arm, args.tier, args.repeat)
    prompt = external.stage(spec, Path(args.dest), args.material_filename)
    (Path(args.dest).parent / f"{Path(args.dest).name}.prompt.md").write_text(prompt)
    print(
        json.dumps(
            {"task_folder": args.dest, "material": str(spec.material) if spec.material else None}
        )
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Grade an external runner's returned folder and write the run record."""

    config = runner.RunConfig.load(Path(args.config)) if args.config else runner.RunConfig({}, {})
    task = _tasks(args)[args.task]
    spec = _spec(args, task, args.arm, args.tier, args.repeat)
    minutes = tuple(int(m) for m in args.checkpoints.split(","))
    record = external.ingest(spec, Path(args.returned), Path(args.out), config, minutes)
    print(json.dumps({k: record[k] for k in ("task", "arm", "tier", "end_reason", "passed")}))
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
    p.add_argument("--tasks", help="limit the environment's guard to these tasks (per-task env)")
    p = add("check-task", cmd_check_task)
    p.add_argument("--task", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--out")
    p.add_argument("--model-cache")
    p = add("prefetch", cmd_prefetch)
    p.add_argument("--cache", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--tasks")
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
    for name, func in (("stage", cmd_stage), ("ingest", cmd_ingest)):
        p = add(name, func)
        p.add_argument("--task", required=True)
        p.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
        p.add_argument("--tier", required=True, choices=runner.TIERS)
        p.add_argument("--repeat", type=int, default=0)
        p.add_argument("--envs", required=True)
        p.add_argument("--material-root", default=str(MATERIAL_ROOT))
        if name == "stage":
            p.add_argument("--dest", required=True)
            p.add_argument("--material-filename", default="AGENTS.md")
        else:
            p.add_argument("--returned", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--config")
            p.add_argument("--checkpoints", default="10,20,45")
    p = add("summarize", cmd_summarize)
    p.add_argument("--results", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
