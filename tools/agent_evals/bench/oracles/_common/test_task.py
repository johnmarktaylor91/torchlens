"""Task test runner: the reference must pass, every wrong solution and the
do-nothing control must fail.

    python _common/test_task.py TASK_DIR [--only NAME ...] [--skip-run]

Runs each solution on TASK_DIR/inputs (and TASK_DIR/heldout when present),
grades with TASK_DIR/grader.py, prints a table and writes
TASK_DIR/.work/results.json. Exit 0 only when every verdict matches.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, **kw):
    print("+", " ".join(os.path.relpath(c) if os.path.isabs(c) else c for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task_dir")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip-run", action="store_true", help="grade existing outputs only")
    a = p.parse_args()
    task = os.path.abspath(a.task_dir)
    meta = json.load(open(os.path.join(task, "task.json")))
    timeout = float(meta.get("timeout_s", 1800))
    work = os.path.join(task, ".work")
    cache = os.path.join(work, "oracle_cache")
    os.makedirs(cache, exist_ok=True)
    heldout = os.path.join(task, "heldout") if os.path.isdir(os.path.join(task, "heldout")) else None

    sols = {"reference": (os.path.join(task, "reference", "solve.py"), True),
            "do_nothing": (os.path.join(HERE, "do_nothing", "solve.py"), False)}
    for w in sorted(glob.glob(os.path.join(task, "wrong", "*", "solve.py"))):
        sols["wrong/" + os.path.basename(os.path.dirname(w))] = (w, False)
    if a.only:
        sols = {k: v for k, v in sols.items() if k in a.only}

    results = {}
    ok = True
    # Reference first so the grader's oracle cache is warm before the wrong ones.
    for name, (solve, expect_pass) in sols.items():
        out = os.path.join(work, name.replace("/", "_"), "outputs")
        hout = os.path.join(work, name.replace("/", "_"), "heldout_outputs") if heldout else None
        if not a.skip_run:
            run([sys.executable, os.path.join(HERE, "run_solution.py"), solve,
                 os.path.join(task, "inputs"), out, "--timeout", str(timeout), "--task-dir", task])
            if heldout:
                run([sys.executable, os.path.join(HERE, "run_solution.py"), solve,
                     heldout, hout, "--timeout", str(timeout), "--task-dir", task])
        cmd = [sys.executable, os.path.join(task, "grader.py"), "--task-dir", task,
               "--outputs", out, "--cache", cache]
        if heldout:
            cmd += ["--heldout-outputs", hout]
        r = run(cmd, capture_output=True, text=True)
        sys.stderr.write(r.stderr[-4000:])
        try:
            rep = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            rep = {"passed": False, "bars": [], "error": r.stdout[-2000:] + r.stderr[-2000:]}
        verdict_ok = (rep["passed"] == expect_pass)
        ok &= verdict_ok
        results[name] = {"expect_pass": expect_pass, "passed": rep["passed"],
                         "verdict_ok": verdict_ok, "bars": rep.get("bars", [])}
    print()
    print(f"== {meta['id']} ==")
    print(f"{'solution':32} {'expected':9} {'got':6} ok   bars")
    for name, r in results.items():
        bars = " ".join(f"{b['name']}={'P' if b['passed'] else 'F'}({b['value']})" for b in r["bars"])
        print(f"{name:32} {'PASS' if r['expect_pass'] else 'FAIL':9} "
              f"{'PASS' if r['passed'] else 'FAIL':6} {'yes' if r['verdict_ok'] else 'NO ':4} {bars}")
    print("TASK", meta["id"], "GREEN" if ok else "RED")
    json.dump(results, open(os.path.join(work, "results.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
