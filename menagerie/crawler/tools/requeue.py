"""Append an explicit bounded effort grant without rewriting crawler history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
)
from menagerie.crawler.driver import DriverLock, DriverLockError


def build_parser() -> argparse.ArgumentParser:
    """Build the requeue tool argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Typed parser for one bounded attempts grant.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stable_id")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--grant", required=True, type=int, help="additional attempt count")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--granted-by", default="operator")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ledger", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate and append one new work generation.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero after a durable append, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    try:
        repo_root = args.repo_root.resolve()
        ledger = args.ledger or (
            repo_root / "menagerie" / "crawler" / "records" / "operational" / "requeue-grants.jsonl"
        )
        owner = {
            "pid": os.getpid(),
            "run_id": "requeue-tool",
            "target": None,
        }
        with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            payload = build_canonical_requeue_grant(
                ledger,
                stable_id=args.stable_id,
                stage=args.stage,
                reason=args.reason,
                attempts=args.grant,
                granted_by=args.granted_by,
            )
            append_canonical_requeue_grant(ledger, payload)
        print(json.dumps(payload, sort_keys=True))
    except (DriverLockError, OSError, ValueError) as exc:
        print(f"requeue failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
