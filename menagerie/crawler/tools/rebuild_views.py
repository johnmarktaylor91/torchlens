"""Rebuild deterministic crawler views from canonical JSONL ledgers only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from menagerie.crawler.authority import AuthorityContext, build_authority_context
from menagerie.crawler.constants import ACCESS_BLOCKED_STATUS_CODE
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.reducer import (
    default_ledger_paths,
    materialize_current,
)
from menagerie.crawler.state import rebuild_state
from menagerie.crawler.status import (
    access_barrier_probes,
    funnel_counts,
    record_is_release_eligible,
)


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


def _access_blocked_row(record: Mapping[str, object]) -> JsonObject:
    """Project one access-blocked model into its recovery worklist row.

    This view exists so that "which models are blocked ONLY on access, and what
    specifically was unreachable" is a file read rather than a re-derivation. The
    campaign runs once; the batch that would recover these models has to be
    assembled from the record months later, and a count alone cannot be actioned.

    Every field is what a later access pass needs to go and fetch the material: the
    exact locator, the registry identity the machine derived from it, when we tried,
    and what the attempt observed. The author's claimed class travels beside the
    machine's own outcome so the disagreement stays visible here too.

    Parameters
    ----------
    record:
        Current terminal model revision with the access-blocked status.

    Returns
    -------
    dict[str, Any]
        Deterministic worklist row for exactly one model.
    """

    resolution = record.get("source_resolution", {})
    search_report = (
        resolution.get("search_report", {}) if isinstance(resolution, Mapping) else {}
    )
    status = record.get("status", {})
    return {
        "stable_id": record.get("stable_id"),
        "status_code": status.get("code") if isinstance(status, Mapping) else None,
        "conclusion": (
            search_report.get("conclusion") if isinstance(search_report, Mapping) else None
        ),
        "barriers": [
            {
                "identifier_kind": probe.get("identifier_kind"),
                "identifier": probe.get("identifier"),
                "locator": probe.get("locator"),
                "attempted_at": probe.get("attempted_at"),
                "probe_outcome": probe.get("probe_outcome"),
                "http_status": probe.get("http_status"),
                "author_claimed_class": probe.get("author_claimed_class"),
            }
            for probe in access_barrier_probes(record)
        ],
    }


def rebuild_views(
    intake: Path,
    records_root: Path,
    views_root: Path,
    database: Path,
    *,
    context: AuthorityContext,
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
    context:
        Mandatory active authority shared by state and view projection.

    Returns
    -------
    dict[str, str]
        View name to complete-file SHA-256 digest.
    """

    ledgers = default_ledger_paths(records_root)
    rebuild_state(database, intake, ledgers, context=context)
    current_by_id = materialize_current(ledgers, context=context)
    current: list[JsonObject] = [current_by_id[key] for key in sorted(current_by_id)]
    release = [record for record in current if record_is_release_eligible(record, current_by_id)]
    deferred = [
        record
        for record in current
        if record["status"]["code"] in {"deferred:needs-cuda", "deferred:needs-x86"}
    ]
    blocked_on_access = [
        _access_blocked_row(record)
        for record in current
        if record["status"]["code"] == ACCESS_BLOCKED_STATUS_CODE
    ]
    digests = {
        "current": _write_jsonl(views_root / "current-models" / "current.jsonl", current),
        "release": _write_jsonl(views_root / "release-models.jsonl", release),
        "deferred": _write_jsonl(views_root / "deferred-linux.jsonl", deferred),
        "blocked_on_access": _write_jsonl(
            views_root / "blocked-on-access.jsonl", blocked_on_access
        ),
    }
    summary = {
        "current_count": len(current),
        "release_count": len(release),
        "deferred_count": len(deferred),
        "blocked_on_access_count": len(blocked_on_access),
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
        snapshot = load_intake_snapshot(args.intake.parent)
        context = build_authority_context(
            active_intake_snapshot_id=snapshot.snapshot_id,
            active_intake_snapshot_sha256=snapshot.snapshot_sha256,
            intake_rows=(item.to_dict() for item in snapshot.items),
            author_model="claude-sonnet",
            author_version="current",
            checker_model="codex",
            checker_version="current",
        )
        digests = rebuild_views(
            args.intake,
            args.records_root,
            args.views_root,
            args.database,
            context=context,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"view rebuild failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"view_digests": digests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
