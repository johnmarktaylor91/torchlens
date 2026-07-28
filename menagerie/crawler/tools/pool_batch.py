"""Batch the author-queue service loop so one managing session can drive a campaign.

The pool loop in ``RUNBOOK_AUTHOR_POOL.md`` is four steps per job: list, claim, dispatch a
subagent, complete. Driven one job at a time that is roughly three shell commands and one
dispatch per model. At 28,482 models it is not operable by hand, and every hand-run step is
a chance to mistype a lease owner or lose a ``claimed_at``.

This tool collapses the mechanical steps into two batch calls, leaving exactly one thing for
the managing session to do -- the part only it can do, dispatching Agent subagents:

    1. ``claim --count N``   -> leases up to N jobs, writes each brief to its own file,
                                emits a manifest
    2. (session dispatches N subagents, one per manifest row, in a single message)
    3. ``complete --manifest`` -> commits every answered job, records honest consumption,
                                and files typed failures for the rest

Design notes that matter:

* **The owner is pinned.** Each shell invocation would otherwise get a fresh ``host:pid``
  owner, so a job claimed in one call cannot be completed by the next -- the lease looks
  foreign. ``--owner`` defaults to a stable value for exactly this reason.
* **Tool-call counts are never invented.** ``complete`` requires a counts file supplying the
  real observed number per job. The engine refuses a receipt declaring more than the grant,
  and the runbook is explicit that rounding a count down to make a job land corrupts the
  effort ledger. A missing count is an error here, not a default.
* **Failures are typed, never guessed.** ``--retryable``/``--permanent`` is required by the
  engine because guessing turns an infrastructure blip into a permanently burned model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

DEFAULT_OWNER = "pool-manager"


def _pool(queue: Path, owner: str, *args: str, repo_root: Path, python: str) -> tuple[int, str, str]:
    """Run one author_pool subcommand and return (rc, stdout, stderr)."""

    argv = [
        python,
        "-m",
        "menagerie.crawler.author_pool",
        "--queue",
        str(queue),
        "--owner",
        owner,
        *args,
    ]
    done = subprocess.run(argv, cwd=repo_root, capture_output=True, text=True)
    return done.returncode, done.stdout, done.stderr


def _json_or_none(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def cmd_claim(args: argparse.Namespace) -> int:
    """Lease up to N pending jobs and write their briefs plus a manifest."""

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = _pool(
        args.queue, args.owner, "list", repo_root=args.repo_root, python=args.python
    )
    if rc != 0:
        sys.stderr.write(stderr or "author-pool list failed\n")
        return 1
    pending = _json_or_none(stdout) or []

    # Skip anything already leased by someone else; a live lease is not ours to take.
    candidates = [
        job
        for job in pending
        if not job.get("leased_by") or job.get("leased_by") == args.owner
    ]

    rows: list[dict[str, Any]] = []
    for job in candidates:
        if len(rows) >= args.count:
            break
        job_id = str(job["job_id"])
        rc, stdout, stderr = _pool(
            args.queue,
            args.owner,
            "claim",
            "--job",
            job_id,
            repo_root=args.repo_root,
            python=args.python,
        )
        if rc != 0:
            # A job that cannot be claimed right now is not an error for the batch --
            # report it and keep going, so one bad row cannot stall a whole round.
            sys.stderr.write(f"skip {job_id}: {(stderr or '').strip()}\n")
            continue
        claimed = _json_or_none(stdout)
        if not isinstance(claimed, dict) or "brief" not in claimed:
            sys.stderr.write(f"skip {job_id}: claim returned no brief\n")
            continue
        brief_path = out / f"{job_id}.brief.md"
        brief_path.write_text(claimed["brief"], encoding="utf-8")
        rows.append(
            {
                "job_id": job_id,
                "kind": claimed.get("kind"),
                "stable_id": claimed.get("stable_id"),
                "subagent_model": claimed.get("subagent_model"),
                "claimed_at": claimed.get("claimed_at"),
                "deadline_at": claimed.get("deadline_at"),
                "required_output_path": claimed.get("required_output_path"),
                "brief_path": str(brief_path),
            }
        )

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"claimed": len(rows), "manifest": str(manifest)}, sort_keys=True))
    for row in rows:
        print(f"  {row['kind']:16s} {row['stable_id']:24s} {row['subagent_model']:8s} {row['brief_path']}")
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    """Commit answered jobs and file typed failures for the rest."""

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    counts = json.loads(Path(args.counts).read_text(encoding="utf-8"))

    ok = 0
    failed = 0
    for row in rows:
        job_id = row["job_id"]
        entry = counts.get(job_id)
        if entry is None:
            sys.stderr.write(f"{job_id}: no entry in counts file -- refusing to guess\n")
            failed += 1
            continue

        # A typed failure was requested for this job.
        if entry.get("failed"):
            reason = entry.get("reason") or "subagent-transient"
            classification = "--retryable" if entry.get("retryable", True) else "--permanent"
            call = [
                "fail",
                "--job",
                job_id,
                "--reason",
                str(reason),
                classification,
            ]
            if entry.get("detail"):
                call += ["--detail", str(entry["detail"])]
            rc, stdout, stderr = _pool(
                args.queue, args.owner, *call, repo_root=args.repo_root, python=args.python
            )
            print(f"{job_id}: fail({reason},{classification[2:]}) rc={rc}")
            failed += 1
            continue

        tool_calls = entry.get("tool_calls")
        if not isinstance(tool_calls, int):
            sys.stderr.write(f"{job_id}: tool_calls missing or not an int -- refusing to guess\n")
            failed += 1
            continue

        call = [
            "complete",
            "--job",
            job_id,
            "--claimed-at",
            str(row["claimed_at"]),
            "--tool-calls",
            str(tool_calls),
        ]
        if entry.get("evidence"):
            call += ["--evidence", str(entry["evidence"])]
        if entry.get("note"):
            call += ["--note", str(entry["note"])]

        rc, stdout, stderr = _pool(
            args.queue, args.owner, *call, repo_root=args.repo_root, python=args.python
        )
        if rc == 0:
            ok += 1
            print(f"{job_id}: complete tool_calls={tool_calls}")
        else:
            failed += 1
            sys.stderr.write(f"{job_id}: complete failed rc={rc}: {(stderr or '').strip()}\n")

    print(json.dumps({"completed": ok, "not_completed": failed}, sort_keys=True))
    return 0 if failed == 0 else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print a compact queue summary."""

    rc, stdout, stderr = _pool(
        args.queue, args.owner, "list", repo_root=args.repo_root, python=args.python
    )
    if rc != 0:
        sys.stderr.write(stderr or "author-pool list failed\n")
        return 1
    pending = _json_or_none(stdout) or []
    by_kind: dict[str, int] = {}
    leased = 0
    for job in pending:
        by_kind[str(job.get("kind"))] = by_kind.get(str(job.get("kind")), 0) + 1
        if job.get("leased_by"):
            leased += 1
    print(json.dumps({"pending": len(pending), "leased": leased, "by_kind": by_kind}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.pool_batch",
        description="Batch the author-queue service loop for one campaign.",
    )
    parser.add_argument("--queue", type=Path, required=True, help="campaign author-queue root")
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="stable lease owner")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="campaign clone root")
    parser.add_argument("--python", default=sys.executable, help="interpreter running the pool")
    sub = parser.add_subparsers(dest="command", required=True)

    claim = sub.add_parser("claim", help="lease N jobs and write their briefs")
    claim.add_argument("--count", type=int, default=12, help="max jobs to lease this round")
    claim.add_argument("--out", type=Path, required=True, help="directory for briefs + manifest")
    claim.set_defaults(func=cmd_claim)

    complete = sub.add_parser("complete", help="commit answered jobs from a manifest")
    complete.add_argument("--manifest", type=Path, required=True)
    complete.add_argument(
        "--counts",
        type=Path,
        required=True,
        help='JSON: {"<job_id>": {"tool_calls": N} | {"failed": true, "reason": "...", "retryable": bool}}',
    )
    complete.set_defaults(func=cmd_complete)

    status = sub.add_parser("status", help="compact queue summary")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
