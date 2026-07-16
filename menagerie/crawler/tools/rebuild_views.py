"""Rebuild deterministic crawler views from canonical JSONL ledgers only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import canonical_json_bytes, hash_bytes
from menagerie.crawler.models import JsonObject
from menagerie.crawler.reducer import default_ledger_paths, materialize_current
from menagerie.crawler.state import rebuild_state
from menagerie.crawler.status import funnel_counts, record_is_release_eligible


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic view-rebuild argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for intake, canonical ledgers, and disposable destinations.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    return parser


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> str:
    """Atomically write sorted JSONL records and return their byte digest.

    Parameters
    ----------
    path:
        Derived view destination.
    records:
        Current records already ordered by stable ID.

    Returns
    -------
    str
        SHA-256 digest of the complete derived file.
    """

    data = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    _atomic_write(path, data)
    return hash_bytes(data)


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace one disposable view atomically.

    Parameters
    ----------
    path:
        Destination path.
    data:
        Complete file bytes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rebuild_views(
    intake: Path, records_root: Path, views_root: Path, database: Path
) -> dict[str, str]:
    """Rebuild current, release, deferred, and summary views.

    Parameters
    ----------
    intake:
        Canonical intake JSONL path.
    records_root:
        Root containing the three canonical ledgers.
    views_root:
        Disposable derived-view root.
    database:
        Disposable SQLite state rebuilt during this operation.

    Returns
    -------
    dict[str, str]
        View name to complete-file SHA-256 digest.
    """

    ledgers = default_ledger_paths(records_root)
    rebuild_state(database, intake, ledgers)
    current_by_id = materialize_current(ledgers)
    current: list[JsonObject] = [current_by_id[key] for key in sorted(current_by_id)]
    release = [record for record in current if record_is_release_eligible(record, current_by_id)]
    deferred = [
        record
        for record in current
        if record["status"]["code"] in {"deferred:needs-cuda", "deferred:needs-x86"}
    ]
    digests = {
        "current": _write_jsonl(views_root / "current-models" / "current.jsonl", current),
        "release": _write_jsonl(views_root / "release-models.jsonl", release),
        "deferred": _write_jsonl(views_root / "deferred-linux.jsonl", deferred),
    }
    summary = {
        "current_count": len(current),
        "release_count": len(release),
        "deferred_count": len(deferred),
        "funnel": funnel_counts(current),
        "view_digests": digests,
    }
    summary_bytes = canonical_json_bytes(summary) + b"\n"
    _atomic_write(views_root / "status-summary.json", summary_bytes)
    digests["status"] = hash_bytes(summary_bytes)
    return digests


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Rebuild views and print their deterministic digests.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero on a complete deterministic rebuild, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    try:
        digests = rebuild_views(args.intake, args.records_root, args.views_root, args.database)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"view rebuild failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"view_digests": digests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
