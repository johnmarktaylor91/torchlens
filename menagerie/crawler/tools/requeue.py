"""Append an explicit bounded effort grant without rewriting crawler history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.effort import EffortGrant
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.recordio import scan_jsonl


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
    parser.add_argument("--ledger", type=Path, default=Path(".crawl-local/requeue-grants.jsonl"))
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
        if args.grant < 1:
            raise ValueError("--grant must be positive")
        prior = scan_jsonl(args.ledger, validate=False)
        generation = len(prior) + 1
        grant_id = stable_hash(
            {
                "generation": generation,
                "stable_id": args.stable_id,
                "stage": args.stage,
                "reason": args.reason,
                "attempts": args.grant,
                "granted_by": args.granted_by,
            }
        )
        grant = EffortGrant(
            grant_id=grant_id,
            stage=args.stage,
            reason=args.reason,
            granted_by=args.granted_by,
            attempts=args.grant,
        )
        payload = {
            **asdict(grant),
            "stable_id": args.stable_id,
            "new_work_generation": generation,
        }
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        with args.ledger.open("ab") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(payload, sort_keys=True))
    except (OSError, ValueError) as exc:
        print(f"requeue failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
