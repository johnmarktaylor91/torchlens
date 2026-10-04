"""Diagnostic-only: run a shard of tests/test_real_world_models.py's slow tier,
one test per process (mirrors run_pytest_per_file_sharded.py / weekly.yml),
and report which ones fail, with their actual failure message.

Not part of the repo; lives in scratch for one-off evidence gathering for
the FJ-weekly-green ledger. Each shard independently collects the full
(stable, -p no:randomly) node id list and takes every --shard-index-th slice
of --num-shards -- no coordinator step needed, every shard agrees on the
same split because collection order is deterministic with randomization off.
Writes a JSON summary to --out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def collect_node_ids(repo: Path, markexpr: str, target: str) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            target,
            "-p",
            "no:randomly",
            "-m",
            markexpr,
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    ids = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if "::" in line and line.split("::", 1)[0].endswith(".py"):
            ids.append(line)
    return ids


def shard(node_ids: list[str], shard_index: int, num_shards: int) -> list[str]:
    """Return this shard's contiguous slice of a stable-ordered id list."""

    total = len(node_ids)
    batch_count = -(-total // num_shards)  # ceil division
    start = shard_index * batch_count
    end = start + batch_count
    return node_ids[start:end]


def run_one(repo: str, node_id: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:randomly", node_id, "--tb=short", "-q", "-rA"],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=1800,
    )
    outcome = "passed" if proc.returncode == 0 else "failed"
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
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    all_node_ids = collect_node_ids(repo, args.marker, args.target)
    my_node_ids = shard(all_node_ids, args.shard_index, args.num_shards)
    print(
        f"shard {args.shard_index}/{args.num_shards}: "
        f"{len(my_node_ids)} of {len(all_node_ids)} node ids",
        flush=True,
    )

    results = []
    for done, node_id in enumerate(my_node_ids, start=1):
        res = run_one(str(repo), node_id)
        results.append(res)
        print(f"[{done}/{len(my_node_ids)}] {res['outcome']:7s} {node_id}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["outcome"] != "passed"]
    print(f"\nshard {args.shard_index}: {len(failed)} failed/crashed of {len(results)}")
    for r in failed:
        print(" -", r["node_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
