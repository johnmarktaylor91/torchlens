"""Read-only ledger reader (F03 item 11): ``tl.experiment.read_ledger(path)``.

Because durability is per-event, the on-disk artifact is FRESH
mid-experiment: an agent reads its own live trajectory through the file with
no live-process socket. Reading never arms the ledger: the reader holds no
ContextVar, writes nothing and never fsyncs. No read runs a model, writes a
verdict or executes an opaque metric; the write door is a named deferral
whose plumbing (actor ids, event ids, append-only amendments, one-writer
rules) already rides the schema. This module owns the SCHEMA
(transport-agnostic, JSON-round-trippable payloads under the
``torchlens.ledger_*_v1`` ids) and the stdio server in
:mod:`torchlens.bridge.mcp` serves the same payloads verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors.episode import BundleExperimentError
from ._ledger import EvidenceRef, LedgerEntry, LedgerEvent, read_ledger_artifact

__all__ = ["read_ledger"]


class _LedgerReader:
    """A read-only view of one saved ``.tlledger`` artifact.

    Built by :func:`read_ledger`; not a public class. The artifact is read
    once when the reader is built (``read_ledger_artifact``: never locks,
    never writes), so a reader is a snapshot of the file at open time. Call
    ``read_ledger(path)`` again for a fresh view of a live experiment.
    """

    __slots__ = ("_entries", "_events", "_path", "_torn")

    def __init__(self, path: str | Path) -> None:
        events, entries, torn = read_ledger_artifact(path)
        self._path = str(path)
        self._events: list[LedgerEvent] = events
        self._entries: dict[str, LedgerEntry] = entries
        self._torn = torn

    def __repr__(self) -> str:
        return (
            f"read_ledger({self._path!r}: {len(self._entries)} entries, {len(self._events)} events)"
        )

    @property
    def overview(self) -> dict[str, Any]:
        """One bounded line per entry (D4h): the agent's working-memory recovery.

        Status, verdict and basis presence, step counts, observation counts
        and quarantine FIRST-LINE-VISIBLE. Payload schema
        ``torchlens.ledger_overview_v1``.
        """
        return {
            "schema": "torchlens.ledger_overview_v1",
            "path": self._path,
            "n_events": len(self._events),
            "torn_tail_discarded": self._torn,
            "entries": [entry.overview_line() for entry in self._entries.values()],
        }

    def entry(self, entry_id: str, *, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        """Paginated trajectory for ONE entry plus EvidenceRef handles.

        Payload schema ``torchlens.ledger_entry_v1``. Refuses
        ``ledger_entry_unknown`` for an id the artifact does not hold.
        """
        if entry_id not in self._entries:
            raise BundleExperimentError(
                f"no ledger entry {entry_id!r} at {self._path}; entries: {sorted(self._entries)}",
                code="ledger_entry_unknown",
                entry_id=entry_id,
            )
        entry_events = [event for event in self._events if event.entry_id == entry_id]
        start = page * page_size
        window = entry_events[start : start + page_size]
        projection = self._entries[entry_id]
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
            "evidence_refs": self._evidence_refs(entry_events),
        }

    def _evidence_refs(self, entry_events: list[LedgerEvent]) -> list[dict[str, Any]]:
        root = Path(self._path).resolve().parent
        refs: list[dict[str, Any]] = []
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
        return refs

    def evidence(self, entry_id: str, index: int = 0) -> dict[str, Any]:
        """Digest-verify ONE evidence ref, then serve the SAME public queries.

        For a bundle artifact ref this loads the bundle and serves its
        ``provenance()`` rows plus the stored effect-table summaries, the
        exact views a live session would read; never a re-graded copy.
        Payload schema ``torchlens.ledger_evidence_v1``; refuses
        ``ledger_evidence_unavailable`` when ``index`` has no ref.
        """
        served = self.entry(entry_id, page=0, page_size=10_000)
        refs = served["evidence_refs"]
        if not refs or index >= len(refs):
            raise BundleExperimentError(
                f"ledger entry {entry_id!r} has {len(refs)} evidence ref(s); "
                f"index {index} is out of range",
                code="ledger_evidence_unavailable",
                entry_id=entry_id,
            )
        ref = refs[index]
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
        root = Path(self._path).resolve().parent
        target = root / ref["uri"]
        from .. import load as tl_load

        bundle = tl_load(str(target))
        provenance_rows = bundle.provenance() if hasattr(bundle, "provenance") else []
        tables = getattr(bundle, "_effect_tables", {}) or {}
        return {
            "schema": "torchlens.ledger_evidence_v1",
            "ref": ref,
            "served": {
                "provenance": [
                    {**row, "lanes": list(row.get("lanes", ()))} for row in provenance_rows
                ],
                "effect_tables": {
                    operation_id: table.to_payload() for operation_id, table in tables.items()
                },
            },
            "disclosure": None,
        }


def read_ledger(path: str | Path) -> _LedgerReader:
    """Open a saved experiment ledger for reading, without arming it.

    ``read_ledger(path).overview`` is the one-line-per-entry summary,
    ``.entry(entry_id, page=0, page_size=50)`` the paginated trajectory of one
    entry and ``.evidence(entry_id, index=0)`` one digest-verified evidence
    ref served through the public bundle queries. Reading never arms the
    ledger: no ContextVar, no writes, no fsync. The payloads are the same
    JSON-round-trippable dicts the MCP tools serve.
    """
    return _LedgerReader(path)
