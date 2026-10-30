"""Run one solution against one inputs directory under a timeout.

    python run_solution.py SOLVE_PY INPUTS OUTPUTS [--timeout S]

Writes OUTPUTS/_run.log and OUTPUTS/_status.json. Exit code is the solution's.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("solve_py")
    p.add_argument("inputs")
    p.add_argument("outputs")
    p.add_argument("--timeout", type=float, default=1800.0)
    a = p.parse_args()
    os.makedirs(a.outputs, exist_ok=True)
    solve = os.path.abspath(a.solve_py)
    cmd = [sys.executable, solve, "--inputs", os.path.abspath(a.inputs),
           "--outputs", os.path.abspath(a.outputs)]
    t0 = time.time()
    status = {"cmd": cmd, "timed_out": False, "returncode": None}
    with open(os.path.join(a.outputs, "_run.log"), "w") as log:
        try:
            r = subprocess.run(cmd, cwd=os.path.dirname(solve), stdout=log,
                               stderr=subprocess.STDOUT, timeout=a.timeout)
            status["returncode"] = r.returncode
        except subprocess.TimeoutExpired:
            status["timed_out"] = True
            status["returncode"] = 124
    status["seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(a.outputs, "_status.json"), "w") as f:
        json.dump(status, f, indent=1)
    print(f"{os.path.relpath(solve)}: rc={status['returncode']} {status['seconds']}s")
    return status["returncode"]


if __name__ == "__main__":
    sys.exit(main())
