"""Diagnostic-only: run every slow test in tests/test_real_world_models.py,
one test per process (mirrors run_pytest_per_file_sharded.py / weekly.yml),
with N worker processes in parallel, and report which ones fail.

Not part of the repo; lives in scratch for one-off evidence gathering for
the FJ-weekly-green ledger. Writes a JSON summary to --out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def collect_node_ids(repo: Path, markexpr: str, target: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-m", markexpr, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    ids = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if "::" in line and line.split("::")[0].endswith(".py"):
            ids.append(line)
    return ids


def run_one(repo: str, node_id: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", node_id, "--tb=short", "-q", "-rA"],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=1800,
    )
    outcome = "passed" if proc.returncode == 0 else "failed"
    # Grab the short summary line(s) and last part of traceback for a failure class.
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    return {
        "node_id": node_id,
        "returncode": proc.returncode,
        "outcome": outcome,
        "tail": tail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--target", default="tests/test_real_world_models.py")
    ap.add_argument("--marker", default="slow and not rare")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    node_ids = collect_node_ids(repo, args.marker, args.target)
    print(f"collected {len(node_ids)} node ids", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, str(repo), nid): nid for nid in node_ids}
        for done, fut in enumerate(as_completed(futures), start=1):
            nid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 diagnostic script
                res = {"node_id": nid, "returncode": -1, "outcome": "crashed", "tail": str(exc)}
            results.append(res)
            print(f"[{done}/{len(node_ids)}] {res['outcome']:7s} {nid}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["outcome"] != "passed"]
    print(f"\n{len(failed)} failed/crashed of {len(results)}")
    for r in failed:
        print(" -", r["node_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
