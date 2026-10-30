"""Drive one subject-agent run under the caps, snapshot checkpoints, grade and record.

A run gets a fresh work folder holding only the task spec (``TASK.md``), the task's public inputs
and the arm's agent material. The oracle directory, held-out inputs, reference solution and
grader stay outside it. The agent command comes from a runtime configuration kept outside the
repository (``RunConfig``): this module knows tiers only as ``small``, ``mid`` and ``frontier``.

While the agent runs, its event stream is read line by line. The run is stopped at the wall-clock
cap, past the substantive tool-call cap or past the token cap. The work folder is copied at each
checkpoint time (10, 20 and 45 minutes by default); a checkpoint the run did not reach takes the
final state. Every checkpoint is graded by the task's deterministic grader, the agent's code is
audited for guarded names, and one JSON record is written whatever happened.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from . import guards, telemetry
from .registry import BenchTask, parse_verdict

TIERS: tuple[str, ...] = ("small", "mid", "frontier")
PREAMBLE = """You are working in the current folder, which is your whole workspace.
The task is in TASK.md, and its inputs are in inputs/. {material_line}

Write your solution as solve.py in this folder. It will be run from this folder as
`python solve.py --inputs INPUTS_DIR --outputs OUTPUTS_DIR`, on inputs/ and again on inputs you
have not seen, so it must compute everything from the inputs it is given. It must create
OUTPUTS_DIR if needed and write only there, the files TASK.md names in the formats it names.

Python is available as `python`, with the packages this environment provides; nothing else can be
installed and there is no network access. You may write, run and test code as you like. Stop when
solve.py is done.
"""


@dataclass(frozen=True)
class Caps:
    """Per-run limits (FINAL-PLAN defaults: 45 minutes, 60 substantive tool calls)."""

    wall_s: float = 45 * 60
    tool_calls: int = 60
    token_cap: int | None = None
    #: Token kinds counted against ``token_cap``; cache reads are reported but not capped.
    token_cap_keys: tuple[str, ...] = ("input_uncached", "input_cache_write", "output")
    checkpoints_s: tuple[float, ...] = (10 * 60, 20 * 60, 45 * 60)


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for subject agents, loaded from a file outside the repository.

    ``agent_cmd`` per tier is an argv list whose items may use ``{prompt_file}``, ``{prompt}``,
    ``{workdir}``, ``{arm_python}``, ``{arm_bin}``, ``{material}`` and ``{guard}``. The command
    must write its event stream as JSON lines on stdout.
    """

    agent_cmd: Mapping[str, tuple[str, ...]]
    caps: Mapping[str, Caps]
    material_filename: str = "AGENTS.md"
    shell_tools: frozenset[str] = frozenset({"Bash"})
    non_substantive_tools: frozenset[str] = frozenset()
    env: Mapping[str, str] = field(default_factory=dict)
    grade_prefix: tuple[str, ...] = ()
    #: Read-only model cache (weights the tasks use); the agent and solve.py run offline.
    model_cache: Path | None = None
    #: Where graders keep their computed oracles between runs, per task.
    oracle_cache: Path | None = None

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        """Read a configuration file (JSON)."""

        raw = json.loads(path.read_text(encoding="utf-8"))
        tiers = raw["tiers"]
        unknown = sorted(set(tiers) - set(TIERS))
        if unknown:
            raise ValueError(f"unknown tiers {unknown}; tiers are {TIERS}")
        caps = {
            tier: Caps(
                **{
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in spec.get("caps", {}).items()
                }
            )
            for tier, spec in tiers.items()
        }
        return cls(
            agent_cmd={tier: tuple(spec["agent_cmd"]) for tier, spec in tiers.items()},
            caps=caps,
            material_filename=raw.get("material_filename", "AGENTS.md"),
            shell_tools=frozenset(raw.get("shell_tools", ["Bash"])),
            non_substantive_tools=frozenset(raw.get("non_substantive_tools", [])),
            env=dict(raw.get("env", {})),
            grade_prefix=tuple(raw.get("grade_prefix", ())),
            model_cache=Path(raw["model_cache"]) if raw.get("model_cache") else None,
            oracle_cache=Path(raw["oracle_cache"]) if raw.get("oracle_cache") else None,
        )


@dataclass(frozen=True)
class RunSpec:
    """What one run is: a task in an arm at a tier, repeat ``repeat``."""

    task: BenchTask
    arm: str
    tier: str
    repeat: int
    arm_python: Path
    grader_python: Path
    material: Path | None = None
    library: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def prepare_work(spec: RunSpec, run_dir: Path, config: RunConfig) -> tuple[Path, Path, list[str]]:
    """Write the work folder, the prompt file and the guard file; return them and the guards."""

    work = run_dir / "work"
    work.mkdir(parents=True)
    (work / "TASK.md").write_text(spec.task.prompt(), encoding="utf-8")
    if spec.task.inputs is not None:
        shutil.copytree(spec.task.inputs, work / "inputs")
    material_line = "There is no agent material file."
    if spec.material is not None:
        shutil.copy2(spec.material, work / config.material_filename)
        material_line = f"Notes on the tools in this environment are in {config.material_filename}."
    prompt_file = run_dir / "prompt.md"
    prompt_file.write_text(PREAMBLE.format(material_line=material_line), encoding="utf-8")
    entries = guards.entries_for(spec.task.task_id, spec.arm, spec.task.natives)
    guards.write_run_config(run_dir / "guard.json", entries)
    return work, prompt_file, entries


def _reader(stream: IO[str], lines: queue.Queue[str | None], sink: IO[str]) -> None:
    for line in stream:
        sink.write(line)
        sink.flush()
        lines.put(line)
    lines.put(None)


def _stop(proc: subprocess.Popen[str], grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace_s)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _snapshot(work: Path, dest: Path) -> Path:
    shutil.copytree(work, dest, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
    return dest


def _cap_hit(stats: telemetry.StreamStats, caps: Caps) -> str | None:
    if stats.substantive_calls > caps.tool_calls:
        return "tool_cap"
    if caps.token_cap is not None:
        used = sum(stats.total_tokens()[key] for key in caps.token_cap_keys)
        if used > caps.token_cap:
            return "token_cap"
    return None


def drive(
    argv: Sequence[str],
    work: Path,
    run_dir: Path,
    env: Mapping[str, str],
    caps: Caps,
    stats: telemetry.StreamStats,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, float, dict[str, Path], int | None]:
    """Run the agent, enforce the caps and take checkpoint snapshots.

    Returns
    -------
    tuple
        End reason, elapsed seconds, snapshots taken while running (by checkpoint label) and the
        agent's exit code (``None`` when it was stopped).
    """

    lines: queue.Queue[str | None] = queue.Queue()
    snapshots: dict[str, Path] = {}
    pending = sorted(caps.checkpoints_s)
    start = clock()
    with (
        (run_dir / "transcript.jsonl").open("w", encoding="utf-8") as sink,
        (run_dir / "agent.stderr").open("w", encoding="utf-8") as err,
    ):
        proc = subprocess.Popen(
            list(argv),
            cwd=work,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        assert proc.stdout is not None
        thread = threading.Thread(target=_reader, args=(proc.stdout, lines, sink), daemon=True)
        thread.start()
        reason, done = "finished", False
        while not done:
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                done = True
            elif line:
                stats.feed(line)
            elapsed = clock() - start
            while pending and elapsed >= pending[0] and pending[0] < caps.wall_s:
                label = checkpoint_label(pending.pop(0))
                snapshots[label] = _snapshot(work, run_dir / "checkpoints" / label)
            hit = _cap_hit(stats, caps) or ("wall_cap" if elapsed >= caps.wall_s else None)
            if hit and not done:
                reason = hit
                _stop(proc)
                done = True
        _stop(proc)
        thread.join(timeout=30)
        while not lines.empty():
            line = lines.get()
            if line:
                stats.feed(line)
    returncode = proc.returncode if reason == "finished" else None
    if reason == "finished" and returncode not in (0, None):
        reason = "agent_error"
    return reason, clock() - start, snapshots, returncode


def checkpoint_label(seconds: float) -> str:
    """Return a checkpoint's label (``t10`` for ten minutes)."""

    return f"t{round(seconds / 60, 2):g}"


def run_solve(
    spec: RunSpec, solution: Path, inputs: Path, outputs: Path, env: Mapping[str, str]
) -> dict[str, Any]:
    """Run ``solve.py`` from ``solution`` on ``inputs``, as the contract says; return exit facts."""

    outputs.mkdir(parents=True, exist_ok=True)
    argv = [str(spec.arm_python), "solve.py", "--inputs", str(inputs), "--outputs", str(outputs)]
    start = time.monotonic()
    log = outputs.parent / f"{outputs.name}.log"
    with log.open("w", encoding="utf-8") as sink:
        try:
            proc = subprocess.run(
                argv,
                cwd=solution,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                timeout=spec.task.solve_timeout_s,
                check=False,
                start_new_session=True,
            )
            returncode: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    return {"returncode": returncode, "seconds": round(time.monotonic() - start, 1)}


def grade(
    spec: RunSpec, solution: Path, out_dir: Path, env: Mapping[str, str], config: RunConfig
) -> dict[str, Any]:
    """Run the solution on the visible (and held-out) inputs, then the task's grader.

    The solution runs with the agent's environment (the arm's interpreter, the guard, offline);
    the grader runs with the grader's interpreter. Returns the normalized verdict, with the
    solution's exit facts under ``solve`` and the grader's exit code under ``grader_exit``.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    if not (solution / "solve.py").is_file():
        verdict: dict[str, Any] = {"passed": False, "bars": [], "error": "solve.py missing"}
    else:
        task = spec.task
        solve = {
            "visible": run_solve(spec, solution, task.inputs or solution, out_dir / "outputs", env)
        }
        heldout_out = None
        if task.heldout is not None:
            heldout_out = out_dir / "heldout_outputs"
            solve["heldout"] = run_solve(spec, solution, task.heldout, heldout_out, env)
        cache = (
            (config.oracle_cache / task.task_id)
            if config.oracle_cache
            else out_dir / "oracle_cache"
        )
        cache.mkdir(parents=True, exist_ok=True)
        argv = task.grade_argv(str(spec.grader_python), out_dir / "outputs", heldout_out, cache)
        try:
            proc = subprocess.run(
                [*config.grade_prefix, *argv],
                cwd=task.root,
                capture_output=True,
                text=True,
                timeout=task.grade_timeout_s,
                check=False,
            )
            (out_dir / "grader.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
            verdict = parse_verdict(proc.stdout)
            verdict["grader_exit"] = proc.returncode
        except subprocess.TimeoutExpired:
            verdict = {"passed": False, "bars": [], "error": "grader timed out"}
        verdict["solve"] = solve
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    return verdict


def agent_env(spec: RunSpec, config: RunConfig, run_dir: Path) -> dict[str, str]:
    """Return the agent's environment: the arm's interpreter first on PATH, the guard file set."""

    arm_bin = str(spec.arm_python.parent)
    base_path = os.environ.get("PATH", "/usr/bin:/bin")
    env = {
        "PATH": f"{arm_bin}:{base_path}",
        "HOME": os.environ.get("HOME", str(run_dir)),
        "LANG": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        guards.ENV_VAR: str(run_dir / "guard.json"),
        "VIRTUAL_ENV": str(spec.arm_python.parent.parent),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        **config.env,
    }
    if config.model_cache is not None:
        env.update(HF_HOME=str(config.model_cache), TORCH_HOME=str(config.model_cache / "torch"))
    return env


def run_one(spec: RunSpec, run_dir: Path, config: RunConfig) -> dict[str, Any]:
    """Run, grade and record one subject-agent run; the record is also written to ``record.json``.

    A harness failure still writes a record (``end_reason`` ``harness_error``), so every run
    counts.
    """

    run_dir.mkdir(parents=True, exist_ok=False)
    caps = config.caps.get(spec.tier, Caps())
    stats = telemetry.StreamStats(
        shell_tools=config.shell_tools, non_substantive=config.non_substantive_tools
    )
    started = datetime.now(timezone.utc).isoformat()
    record = _empty_record(spec, started, run_dir)
    if not spec.task.applies_to(spec.arm):
        record["end_reason"] = "not_applicable"
        return _write(record, run_dir)
    try:
        work, prompt_file, entries = prepare_work(spec, run_dir, config)
        values = {
            "prompt_file": str(prompt_file),
            "prompt": prompt_file.read_text(encoding="utf-8"),
            "workdir": str(work),
            "arm_python": str(spec.arm_python),
            "arm_bin": str(spec.arm_python.parent),
            "material": config.material_filename,
            "guard": str(run_dir / "guard.json"),
        }
        argv = [part.format(**values) for part in config.agent_cmd[spec.tier]]
        reason, elapsed, snapshots, returncode = drive(
            argv, work, run_dir, agent_env(spec, config, run_dir), caps, stats
        )
        _finish(record, spec, run_dir, work, caps, snapshots, entries, stats, config)
        record.update(end_reason=reason, elapsed_s=round(elapsed, 1), agent_returncode=returncode)
    except Exception as exc:  # noqa: BLE001 - a harness failure is recorded, never dropped
        record["end_reason"] = "harness_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["passed"] = False
    return _write(record, run_dir)


def _finish(
    record: dict[str, Any],
    spec: RunSpec,
    run_dir: Path,
    work: Path,
    caps: Caps,
    snapshots: dict[str, Path],
    entries: list[str],
    stats: telemetry.StreamStats,
    config: RunConfig,
) -> None:
    for seconds in caps.checkpoints_s:
        label = checkpoint_label(seconds)
        if label not in snapshots:
            snapshots[label] = _snapshot(work, run_dir / "checkpoints" / label)
    env = agent_env(spec, config, run_dir)
    record["checkpoints"] = {
        label: grade(spec, path, run_dir / "grades" / label, env, config)
        for label, path in sorted(snapshots.items(), key=lambda item: float(item[0][1:]))
    }
    final_label = checkpoint_label(max(caps.checkpoints_s))
    provided = set()
    if spec.task.inputs is not None:
        provided = {
            f"inputs/{p.relative_to(spec.task.inputs).as_posix()}"
            for p in spec.task.inputs.rglob("*.py")
        }
    files = telemetry.agent_python_files(work, exclude=provided)
    sources = [path.read_text(encoding="utf-8", errors="replace") for path in files]
    violations = sorted({hit for source in sources for hit in guards.audit_source(source, entries)})
    record["guard"] = {"entries": entries, "violations": violations}
    record["load_bearing_lines"] = sum(telemetry.load_bearing_lines(source) for source in sources)
    if spec.library:
        record["first_library_call"] = telemetry.first_library_call(
            [*stats.code_snippets, *sources], spec.library
        )
    record["tokens"] = stats.total_tokens()
    record["tool_calls"] = {
        "total": sum(stats.tool_counts.values()),
        "substantive": stats.substantive_calls,
        "by_name": dict(stats.tool_counts),
    }
    record["executions"] = stats.executions
    record["stream_bad_lines"] = stats.bad_lines
    record["passed"] = bool(record["checkpoints"][final_label]["passed"]) and not violations


def _empty_record(spec: RunSpec, started: str, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": telemetry.SCHEMA_VERSION,
        "run_id": spec.run_id,
        "task": spec.task.task_id,
        "arm": spec.arm,
        "tier": spec.tier,
        "repeat": spec.repeat,
        "started_at": started,
        "elapsed_s": 0.0,
        "end_reason": "harness_error",
        "passed": False,
        "tokens": dict.fromkeys(telemetry.TOKEN_KEYS, 0),
        "tool_calls": {"total": 0, "substantive": 0, "by_name": {}},
        "executions": 0,
        "load_bearing_lines": 0,
        "checkpoints": {},
        "guard": {"entries": [], "violations": []},
        "first_library_call": None,
        "environment": dict(spec.environment),
        "artifacts": {
            "transcript": "transcript.jsonl",
            "code": "work",
            "checkpoints": "checkpoints",
            "grades": "grades",
        },
    }


def _write(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    problems = telemetry.validate_record(record)
    if problems:
        record["schema_problems"] = problems
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record
