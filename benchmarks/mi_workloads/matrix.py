"""Run the (tool x workload) matrix for one model, each cell in a fresh process, then check exactness.

python -m benchmarks.mi_workloads.matrix --model gpt2 --out r.jsonl --resdir res/ [--tools a,b] [--workloads x,y]
Exactness: max abs diff of each tool's result tensor against the hooks result (bit-identical = 0.0);
generation results compare token ids and report the number of differing positions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from .run import WORKLOADS

TOOLS = [
    "hooks",
    "tl_bind",
    "tl_record",
    "tl_trace",
    "tl_trace_default",
    "tl_fast_rerun",
    "nnsight",
    "tlens",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resdir", required=True)
    p.add_argument("--tools", default=",".join(TOOLS))
    p.add_argument("--workloads", default=",".join(WORKLOADS))
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--label", default="")
    p.add_argument("extra", nargs="*")
    a = p.parse_args()
    tools = a.tools.split(",")
    wls = a.workloads.split(",")
    os.makedirs(a.resdir, exist_ok=True)
    for wl in wls:
        for tool in tools:
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.mi_workloads.run",
                "--model",
                a.model,
                "--tool",
                tool,
                "--workload",
                wl,
                "--out",
                a.out,
                "--resdir",
                a.resdir,
                "--label",
                a.label,
                *a.extra,
            ]
            t0 = time.perf_counter()
            try:
                r = subprocess.run(cmd, timeout=a.timeout, capture_output=True, text=True)
                tail = (r.stdout + r.stderr)[-2000:]
                status = "ok" if r.returncode == 0 else f"rc={r.returncode}"
            except subprocess.TimeoutExpired as e:
                tail = (
                    (e.stdout or b"").decode(errors="replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )[-2000:]
                status = "timeout"
            rec = {
                "kind": "cell",
                "model": a.model,
                "tool": tool,
                "workload": wl,
                "status": status,
                "wall_s": time.perf_counter() - t0,
            }
            if status != "ok":
                rec["tail"] = tail
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec)[:400], flush=True)
    # exactness
    for wl in wls:
        ref_path = os.path.join(a.resdir, f"{a.model}.{wl}.hooks.pt")
        if not os.path.exists(ref_path):
            continue
        ref = torch.load(ref_path)
        for tool in tools:
            path = os.path.join(a.resdir, f"{a.model}.{wl}.{tool}.pt")
            if tool == "hooks" or not os.path.exists(path):
                continue
            got = torch.load(path)
            rec = {"kind": "exact", "model": a.model, "tool": tool, "workload": wl}
            if got.shape != ref.shape:
                rec["shape_mismatch"] = [list(got.shape), list(ref.shape)]
            elif got.dtype == torch.int64:
                rec["token_mismatches"] = int((got != ref).sum())
                rec["identical_tokens"] = bool((got == ref).all())
            else:
                rec["max_abs_diff"] = float((got.float() - ref.float()).abs().max())
                rec["ref_abs_max"] = float(ref.float().abs().max())
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
