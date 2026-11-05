"""Runs driven by an external subject runner: stage the task folder, then ingest what comes back.

When the subject agent runs under a separate isolating runner (its own sandbox, login and caps),
the harness does not drive the process. It stages the run's task folder and prompt before, and
afterwards turns the runner's returned folder into the same record ``runner.run_one`` writes.
Grading happens here, in a job with no agent login.

The returned folder holds ``agent.jsonl`` (the event stream), ``record.json`` (the runner's exit
reason, return code and elapsed seconds), ``task/`` (the final task folder) and
``checkpoints/<n>m/`` (snapshots at the requested minutes).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import guards, runner, telemetry

#: The external runner's exit reasons mapped to the record's end reasons.
EXIT_REASONS = {
    "completed": "finished",
    "wall_cap": "wall_cap",
    "tool_cap": "tool_cap",
    "tool_calls": "tool_cap",
    "turn_cap": "tool_cap",
    "turns": "tool_cap",
    "output_token_cap": "token_cap",
    "output_tokens": "token_cap",
    "token_cap": "token_cap",
}


def stage(spec: runner.RunSpec, dest: Path, material_filename: str = "AGENTS.md") -> str:
    """Write the run's task folder at ``dest`` and return the prompt to hand the runner.

    The folder holds ``TASK.md``, ``inputs/`` and the arm's material file, nothing else.
    """

    dest.mkdir(parents=True)
    (dest / "TASK.md").write_text(spec.task.prompt(), encoding="utf-8")
    if spec.task.inputs is not None:
        shutil.copytree(spec.task.inputs, dest / "inputs")
    material_line = "There is no agent material file."
    if spec.material is not None:
        shutil.copy2(spec.material, dest / material_filename)
        material_line = f"Notes on the tools in this environment are in {material_filename}."
    return runner.PREAMBLE.format(material_line=material_line)


def end_reason(raw: dict[str, Any]) -> str:
    """Map the runner's record to an end reason (unknown or credential reasons are harness errors)."""

    reason = EXIT_REASONS.get(str(raw.get("exit_reason")), "harness_error")
    if reason == "finished" and raw.get("returncode") not in (0, None):
        return "agent_error"
    return reason


def ingest(
    spec: runner.RunSpec,
    returned: Path,
    out: Path,
    config: runner.RunConfig,
    checkpoints_min: tuple[int, ...] = (10, 20, 45),
) -> dict[str, Any]:
    """Grade a returned run and write its record to ``out/record.json``.

    A checkpoint the runner did not write (the run ended first) takes the final task folder,
    as in ``runner.run_one``.
    """

    out.mkdir(parents=True, exist_ok=False)
    raw = json.loads((returned / "record.json").read_text(encoding="utf-8"))
    record = runner._empty_record(spec, datetime.now(timezone.utc).isoformat(), out)
    record["artifacts"] = {
        "transcript": str(returned / "agent.jsonl"),
        "code": str(returned / "task"),
        "checkpoints": str(returned / "checkpoints"),
        "grades": "grades",
    }
    record["runner"] = {
        k: raw.get(k) for k in ("exit_reason", "returncode", "elapsed_s", "tool_calls")
    }
    stats = telemetry.StreamStats(
        shell_tools=config.shell_tools, non_substantive=config.non_substantive_tools
    )
    transcript = returned / "agent.jsonl"
    if transcript.exists():
        for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
            stats.feed(line)
    final = returned / "task"
    entries = guards.entries_for(spec.task.task_id, spec.arm, spec.task.natives)
    env = runner.agent_env(spec, config, out)
    guards.write_run_config(out / "guard.json", entries)
    checkpoints = {}
    for minute in sorted(checkpoints_min):
        snapshot = returned / "checkpoints" / f"{minute}m"
        source = snapshot if snapshot.is_dir() else final
        checkpoints[f"t{minute}"] = runner.grade(
            spec, source, out / "grades" / f"t{minute}", env, config
        )
    record["checkpoints"] = checkpoints
    runner.summarize_code(record, spec, final, entries, stats)
    record["end_reason"] = end_reason(raw)
    record["elapsed_s"] = float(raw.get("elapsed_s") or 0.0)
    last = f"t{max(checkpoints_min)}"
    record["passed"] = bool(checkpoints[last]["passed"]) and not record["guard"]["violations"]
    return runner._write(record, out)
