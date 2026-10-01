"""Read-only ledger serving tools (F03 item 11): overview / entry / evidence.

Because durability is per-event, the on-disk artifact is FRESH
mid-experiment: an agent reads its own live trajectory through the file with
no live-process socket. No tool runs a model, arms a process, writes a
verdict, or executes an opaque metric; the write door is a named deferral
whose plumbing (actor ids, event ids, append-only amendments, one-writer
rules) already rides the schema. The tool shapes are handed to the TRI-AGENT
panel; this module owns the SCHEMA (transport-agnostic, JSON-round-trippable)
and the stdio server in :mod:`torchlens.bridge.mcp` wraps these functions.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors.episode import BundleExperimentError
from ._ledger import EvidenceRef, read_ledger_artifact

__all__ = ["ledger_entry", "ledger_evidence", "ledger_overview"]


def ledger_overview(path: str) -> dict[str, Any]:
    """One bounded line per entry (D4h): the agent's working-memory recovery.

    Status, verdict + basis presence, step counts, observation counts, and
    quarantine FIRST-LINE-VISIBLE.
    """

    events, entries, torn = read_ledger_artifact(path)
    return {
        "schema": "torchlens.ledger_overview_v1",
        "path": str(path),
        "n_events": len(events),
        "torn_tail_discarded": torn,
        "entries": [entry.overview_line() for entry in entries.values()],
    }


def ledger_entry(path: str, entry_id: str, *, page: int = 0, page_size: int = 50) -> dict[str, Any]:
    """Paginated trajectory for ONE entry + EvidenceRef handles."""

    events, entries, _torn = read_ledger_artifact(path)
    if entry_id not in entries:
        raise BundleExperimentError(
            f"no ledger entry {entry_id!r} at {path}; entries: {sorted(entries)}",
            code="ledger_entry_unknown",
            entry_id=entry_id,
        )
    entry_events = [event for event in events if event.entry_id == entry_id]
    start = page * page_size
    window = entry_events[start : start + page_size]
    refs: list[dict[str, Any]] = []
    root = Path(path).resolve().parent
    for event in entry_events:
        event_refs = list(event.payload.get("refs", []) or [])
        single = event.payload.get("ref")
        if single:
            event_refs.append(single)  # evidence_relinked carries ONE ref
        for payload_ref in event_refs:
            ref = EvidenceRef(
                kind=str(payload_ref.get("kind", "artifact")),
                uri=str(payload_ref.get("uri", "")),
                object_id=payload_ref.get("object_id"),
                digest=payload_ref.get("digest"),
            )
            refs.append({**ref.to_payload(), "availability": ref.availability(root=root)})
    projection = entries[entry_id]
    return {
        "schema": "torchlens.ledger_entry_v1",
        "entry": projection.overview_line(),
        "hypothesis": projection.hypothesis,
        "hypothesis_declared_at_seq": projection.hypothesis_declared_at_seq,
        "metric": projection.metric,
        "metric_declared_at_seq": projection.metric_declared_at_seq,
        "verdict": projection.verdict,
        "page": page,
        "n_events": len(entry_events),
        "events": [
            {
                "seq": event.seq,
                "kind": event.kind,
                "actor": event.actor,
                "payload": event.payload,
            }
            for event in window
        ],
        "evidence_refs": refs,
    }


def ledger_evidence(path: str, entry_id: str, ref_index: int = 0) -> dict[str, Any]:
    """Digest-verify ONE evidence ref, then serve the SAME public queries.

    For a bundle artifact ref this loads the bundle and serves its
    ``provenance()`` rows plus the stored effect-table summaries — the exact
    views a live session would read; never a re-graded copy.
    """

    served = ledger_entry(path, entry_id, page=0, page_size=10_000)
    refs = served["evidence_refs"]
    if not refs or ref_index >= len(refs):
        raise BundleExperimentError(
            f"ledger entry {entry_id!r} has {len(refs)} evidence ref(s); "
            f"index {ref_index} is out of range",
            code="ledger_evidence_unavailable",
            entry_id=entry_id,
        )
    ref = refs[ref_index]
    if ref["availability"] in ("missing", "stale"):
        # The semantic record outlives its tensors by design: missing/stale
        # evidence stays a queryable historical record, never a crash.
        return {
            "schema": "torchlens.ledger_evidence_v1",
            "ref": ref,
            "served": None,
            "disclosure": (
                f"evidence is {ref['availability']}: the digest recorded at "
                "step time does not resolve to a live artifact (a bundle "
                "overwritten at the same path reads stale, never silently "
                "relinked; relink() is the explicit administrative event)."
            ),
        }
    root = Path(path).resolve().parent
    target = root / ref["uri"]
    from .. import load as tl_load

    bundle = tl_load(str(target))
    provenance_rows = bundle.provenance() if hasattr(bundle, "provenance") else []
    tables = getattr(bundle, "_effect_tables", {}) or {}
    return {
        "schema": "torchlens.ledger_evidence_v1",
        "ref": ref,
        "served": {
            "provenance": [{**row, "lanes": list(row.get("lanes", ()))} for row in provenance_rows],
            "effect_tables": {
                operation_id: table.to_payload() for operation_id, table in tables.items()
            },
        },
        "disclosure": None,
    }
