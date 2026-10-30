"""Run one solution against one inputs directory under a timeout, in an
isolated workspace outside the task tree.

    python run_solution.py SOLVE_PY INPUTS OUTPUTS [--timeout S] [--task-dir DIR]

The inputs are copied into a fresh temporary directory, the solution directory
is copied beside them (together with the task's `reference/` directory when the
solution lives inside the task tree, so the seeded wrong solutions can import
it), and the solution runs with that copy as its working directory. It is
given only the copied inputs and a temporary outputs directory; nothing in the
task tree (held-out inputs, grader, oracle cache) is reachable by a relative
path. The harness that runs subject agents must add real filesystem isolation
on top of this; this runner only keeps the task tree out of the solution's way.

Writes OUTPUTS/_run.log and OUTPUTS/_status.json. Exit code is the solution's.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("solve_py")
    p.add_argument("inputs")
    p.add_argument("outputs")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--task-dir", default=None,
                   help="task directory; its reference/ is copied for wrong-solution imports")
    a = p.parse_args()
    solve = os.path.abspath(a.solve_py)
    sol_dir = os.path.dirname(solve)
    inputs = os.path.abspath(a.inputs)
    outputs = os.path.abspath(a.outputs)
    os.makedirs(outputs, exist_ok=True)
    task_dir = os.path.abspath(a.task_dir) if a.task_dir else None
    if task_dir is None:
        cand = os.path.dirname(os.path.dirname(inputs)) if os.path.basename(inputs) in ("inputs", "heldout") else None
        task_dir = os.path.dirname(inputs) if cand is None else os.path.dirname(inputs)
    status = {"timed_out": False, "returncode": None}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="oracle-run-") as tmp:
        tmp_inputs = os.path.join(tmp, "inputs")
        tmp_outputs = os.path.join(tmp, "outputs")
        shutil.copytree(inputs, tmp_inputs)
        os.makedirs(tmp_outputs)
        # Solution copy, keeping its path relative to the task dir when inside it.
        if task_dir and sol_dir.startswith(task_dir + os.sep):
            rel = os.path.relpath(sol_dir, task_dir)
            tmp_sol = os.path.join(tmp, "solution", rel)
            ref = os.path.join(task_dir, "reference")
            if os.path.isdir(ref) and rel != "reference":
                shutil.copytree(ref, os.path.join(tmp, "solution", "reference"))
        else:
            tmp_sol = os.path.join(tmp, "solution", "main")
        shutil.copytree(sol_dir, tmp_sol, ignore=shutil.ignore_patterns("__pycache__"))
        cmd = [sys.executable, os.path.join(tmp_sol, os.path.basename(solve)),
               "--inputs", tmp_inputs, "--outputs", tmp_outputs]
        status["cmd"] = cmd
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with open(os.path.join(outputs, "_run.log"), "w") as log:
            try:
                r = subprocess.run(cmd, cwd=tmp_sol, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=a.timeout, env=env)
                status["returncode"] = r.returncode
            except subprocess.TimeoutExpired:
                status["timed_out"] = True
                status["returncode"] = 124
        for name in os.listdir(tmp_outputs):
            src = os.path.join(tmp_outputs, name)
            dst = os.path.join(outputs, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    status["seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(outputs, "_status.json"), "w") as f:
        json.dump(status, f, indent=1)
    print(f"{os.path.relpath(solve)}: rc={status['returncode']} {status['seconds']}s")
    return status["returncode"]


if __name__ == "__main__":
    sys.exit(main())
