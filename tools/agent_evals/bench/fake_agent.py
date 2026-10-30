"""A scripted stand-in for a subject agent, for harness tests and model-free smoke runs.

It writes the event stream a headless coding agent writes (JSON lines on stdout) and acts on
the work folder, so the whole pipeline (environment, guard, runner, grader, telemetry) runs with
no model call. Modes:

``solve``     copy ``--solution`` to ``solution.py`` in the work folder and run it once
``nothing``   end without writing anything
``loop``      keep making tool calls (``--calls``) with ``--sleep`` seconds between them
``import``    write and run a script that imports ``--module`` (a guard check)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def emit(event: dict) -> None:
    """Write one event line."""

    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def tool_call(index: int, name: str, tool_input: dict, usage: dict | None = None) -> None:
    """Emit one assistant message holding one tool call."""

    emit(
        {
            "type": "assistant",
            "message": {
                "id": f"msg_{index}",
                "content": [
                    {"type": "tool_use", "id": f"tool_{index}", "name": name, "input": tool_input}
                ],
                "usage": usage
                or {"input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 50},
            },
        }
    )


def run_python(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``python script`` from PATH (the arm's interpreter), as the agent's shell would."""

    return subprocess.run(["python", script], capture_output=True, text=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("solve", "nothing", "loop", "import"), default="solve")
    parser.add_argument("--solution")
    parser.add_argument("--module", default="")
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    emit({"type": "system", "subtype": "init"})
    tool_call(0, "Read", {"file_path": "TASK.md"})
    if args.mode == "solve":
        source = Path(args.solution).read_text(encoding="utf-8")
        tool_call(1, "Write", {"file_path": "solution.py", "content": source})
        shutil.copy2(args.solution, "solution.py")
        tool_call(2, "Bash", {"command": "python solution.py"})
        proc = run_python("solution.py")
        emit(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": proc.stderr[-500:]}]},
            }
        )
    elif args.mode == "import":
        Path("probe.py").write_text(f"import {args.module}\n", encoding="utf-8")
        tool_call(1, "Bash", {"command": "python probe.py"})
        proc = run_python("probe.py")
        Path("probe.log").write_text(proc.stderr, encoding="utf-8")
    elif args.mode == "loop":
        for index in range(args.calls):
            tool_call(10 + index, "Bash", {"command": "ls"})
            time.sleep(args.sleep)
    emit(
        {
            "type": "result",
            "subtype": "success",
            "usage": {
                "input_tokens": 300,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 200,
                "output_tokens": 150,
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
