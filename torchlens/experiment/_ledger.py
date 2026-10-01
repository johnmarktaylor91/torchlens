"""The persistent experiment ledger (F03 items 9-10): event-sourced, chained.

The SEMANTIC record of an experiment — hypothesis, metric CHOICE, verdict —
opt-in, beside the always-on material record (the Bundle), referencing it
and never duplicating it. The third instance of a pattern the repo owns
twice (``distributed/_ledger.py`` is the declared pattern-of-record,
``capture/_episode_ledger.py`` the worked instance): closed vocabularies,
frozen payload key sets, fail-closed loading, write-once rows,
disclosure-never-authority.

THE THREE-LAYER HONESTY LAW (D4c): every field is exactly one of MECHANICAL
(engine-derived, never user-writable), INTERPRETIVE (question, hypothesis,
metric choice, note, verdict — always actor-stamped), or DERIVED
(projections, counts). An interpretive field may never render as a
mechanical one: a verdict carries ``basis`` step ids, and an empty basis
renders "asserted; no recorded basis" — legal, and visibly weaker.
``declared_at_seq`` makes declaration order a checkable fact: TorchLens
never calls a post-hoc metric pre-registered. TorchLens may LINT (a verdict
over zero recorded observations gets a typed disclosure event) and never
blocks or computes a verdict.

PERSISTENCE (D4f): one append-only JSONL artifact — line 1 is the fail-closed
manifest, every later line one hash-chained event (``seq``, ``prev_digest``,
``event_digest``), flush + fsync before an event is reported finalized.
Kill -9 after event k recovers exactly k finalized events; a torn tail line
is discarded AND disclosed; interior tampering refuses at the break.
Contents are REFERENCES ONLY (EvidenceRefs with digests and availability) —
never tensors, prompts, token ids, audit copies, or executable callables.

ARMING (D4d): ContextVar-scoped (never a module global — threads and
unrelated asyncio tasks must not cross-write; async children inherit),
opt-in, off by default, notebook-first: ``ledger(path, hypothesis=,
metric=)`` arms immediately and opens entry 1 (visibly DRAFT without a
hypothesis); the with-form is sugar. Nested different-scope arming refuses
typed; same-scope re-entry is idempotent; ONE writer per artifact in v1.

WRITE FAILURES (D4g, the adjudicated 2-1 ruling): arm-time preflight fails
closed BEFORE any material work; a durable ``material_step_started`` event
precedes material work; a finalization failure AFTER material work warns
exactly once, quarantines and disarms the entry, best-effort writes
``entry_quarantined``, and RETURNS the material result unharmed;
``save()``/``close()`` raise; ``on_record_error="raise"`` is the strict
opt-in, visible in the entry header from the start.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .._io._json import loads_bounded
from ..errors import TorchLensWarning
from ..errors.episode import BundleExperimentError

__all__ = [
    "LEDGER_EVENT_KINDS",
    "VERDICT_TOKENS",
    "EvidenceRef",
    "ExperimentLedger",
    "LedgerEntry",
    "LedgerEvent",
    "active_ledger",
    "ledger",
]

LEDGER_SCHEMA = "torchlens.experiment_ledger_v1"

#: Closed event vocabulary (D4b). Amendments APPEND; nothing rewrites.
LEDGER_EVENT_KINDS = frozenset(
    {
        "entry_opened",
        "hypothesis_set",
        "metric_chosen",
        "material_step_started",
        "material_step_finalized",
        "observation_recorded",
        "verdict_set",
        "note",
        "evidence_relinked",
        "entry_closed",
        "entry_quarantined",
        "lint",
    }
)

#: Closed verdict vocabulary; free text rides BESIDE, never a growing enum.
VERDICT_TOKENS = frozenset({"supports", "refutes", "inconclusive", "not_assessed"})

_EVIDENCE_AVAILABILITY = ("live", "persisted", "missing", "stale")


def _canonical(payload: dict[str, Any]) -> str:
    """Canonical compact JSON for digesting and stream lines (sorted keys)."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class EvidenceRef:
    """One reference to material evidence — never the evidence itself."""

    kind: str
    uri: str
    object_id: str | None = None
    digest: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize this ref to its stream-line payload dict."""

        return {
            "kind": self.kind,
            "uri": self.uri,
            "object_id": self.object_id,
            "digest": self.digest,
        }

    def availability(self, *, root: Path) -> str:
        """live | persisted | missing | stale, computed at READ time."""

        target = (root / self.uri) if not os.path.isabs(self.uri) else Path(self.uri)
        if not target.exists():
            return "missing"
        if self.digest is not None:
            current = _artifact_digest(target)
            if current != self.digest:
                # A bundle overwritten at the same path reads STALE, never
                # silently relinked.
                return "stale"
        return "persisted"


def _artifact_digest(path: Path) -> str:
    """Cheap deterministic digest of an artifact path (identity, not bytes).

    For directory artifacts (``.tlspec`` bundles) the digest covers the
    manifest + bundle metadata files, which carry the identity fields
    (``bundle_id``, member tree hashes) — sufficient to detect an overwrite
    at the same path without hashing tensor blobs.
    """

    digest = hashlib.sha256()
    if path.is_dir():
        for name in ("manifest.json", "bundle.json"):
            candidate = path / name
            if candidate.exists():
                digest.update(candidate.read_bytes())
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LedgerEvent:
    """One append-only, hash-chained ledger event."""

    seq: int
    event_id: str
    entry_id: str
    kind: str
    actor: str
    payload: dict[str, Any]
    prev_digest: str | None

    def __post_init__(self) -> None:
        if self.kind not in LEDGER_EVENT_KINDS:
            raise ValueError(
                f"ledger event kind {self.kind!r} is outside the closed "
                f"vocabulary {sorted(LEDGER_EVENT_KINDS)}"
            )

    @property
    def event_digest(self) -> str:
        """sha256 over the canonical event body incl. prev_digest (the chain link)."""

        return hashlib.sha256(
            _canonical(
                {
                    "seq": self.seq,
                    "event_id": self.event_id,
                    "entry_id": self.entry_id,
                    "kind": self.kind,
                    "actor": self.actor,
                    "payload": self.payload,
                    "prev_digest": self.prev_digest,
                }
            ).encode("utf-8")
        ).hexdigest()

    def to_line(self) -> str:
        """Serialize this event to one canonical JSONL stream line."""

        record = {
            "seq": self.seq,
            "event_id": self.event_id,
            "entry_id": self.entry_id,
            "kind": self.kind,
            "actor": self.actor,
            "payload": self.payload,
            "prev_digest": self.prev_digest,
            "event_digest": self.event_digest,
        }
        return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def from_line_payload(cls, record: dict[str, Any]) -> LedgerEvent:
        """Rebuild an event from a parsed stream line, verifying its stored digest."""

        event = cls(
            seq=int(record["seq"]),
            event_id=str(record["event_id"]),
            entry_id=str(record["entry_id"]),
            kind=str(record["kind"]),
            actor=str(record["actor"]),
            payload=dict(record["payload"]),
            prev_digest=record["prev_digest"],
        )
        if event.event_digest != record.get("event_digest"):
            raise ValueError(f"ledger event seq {event.seq} digest mismatch (interior tamper)")
        return event


@dataclass
class LedgerEntry:
    """DERIVED projection of one entry's events (never authority)."""

    entry_id: str
    question: str | None = None
    status: str = "open"  # open | closed | quarantined
    hypothesis: str | None = None
    hypothesis_declared_at_seq: int | None = None
    metric: str | None = None
    metric_declared_at_seq: int | None = None
    verdict: dict[str, Any] | None = None
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    quarantine_reason: str | None = None

    @property
    def draft(self) -> bool:
        """Visibly DRAFT until a hypothesis is declared."""

        return self.hypothesis is None

    def overview_line(self) -> str:
        """One bounded line (D4h): status, verdict + basis presence, counts."""

        verdict = "verdict=none"
        if self.verdict is not None:
            basis = self.verdict.get("basis") or ()
            basis_note = f"basis={len(basis)}" if basis else "asserted; no recorded basis"
            verdict = f"verdict={self.verdict['token']} ({basis_note})"
        finalized = sum(1 for step in self.steps.values() if step.get("finalized"))
        interrupted = len(self.steps) - finalized
        quarantine = " QUARANTINED" if self.status == "quarantined" else ""
        draft = " DRAFT" if self.draft else ""
        return (
            f"[{self.entry_id}]{quarantine}{draft} status={self.status} {verdict} "
            f"steps={finalized}+{interrupted}i obs={len(self.observations)} "
            f"q={self.question or '-'}"
        )


def _apply_entry_opened(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project entry_opened: stamp the entry's question."""

    entry.question = event.payload.get("question")


def _apply_hypothesis_set(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project hypothesis_set: latest text wins, declaration seq is kept."""

    # Amendments append; the current projection takes the LATEST but keeps
    # the original declaration seq (declaration order is fact).
    if entry.hypothesis is None:
        entry.hypothesis_declared_at_seq = event.seq
    entry.hypothesis = event.payload.get("hypothesis")


def _apply_metric_chosen(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project metric_chosen: latest metric wins, declaration seq is kept."""

    entry.metric = event.payload.get("metric")
    if entry.metric_declared_at_seq is None:
        entry.metric_declared_at_seq = event.seq


def _apply_step_started(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project step_started: open the step row with its operation and refs."""

    payload = event.payload
    entry.steps[str(payload["step_id"])] = {
        "started": True,
        "finalized": False,
        "operation": payload.get("operation"),
        "refs": payload.get("refs", []),
    }


def _apply_step_finalized(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project step_finalized: mark the step done with its outcome and refs."""

    payload = event.payload
    step = entry.steps.setdefault(str(payload["step_id"]), {"started": False, "refs": []})
    step["finalized"] = True
    step["outcome"] = payload.get("outcome")
    step["refs"] = payload.get("refs", step.get("refs", []))


def _apply_observation(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project observation: append the recorded observation payload."""

    entry.observations.append(dict(event.payload))


def _apply_note(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project note: append the free-text note payload."""

    entry.notes.append(dict(event.payload))


def _apply_verdict_set(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project verdict_set: latest verdict wins, actor and seq stamped."""

    payload = event.payload
    entry.verdict = {
        "token": payload.get("token"),
        "text": payload.get("text"),
        "basis": tuple(payload.get("basis", ())),
        "actor": event.actor,
        "at_seq": event.seq,
        "amends": payload.get("amends"),
    }


def _apply_entry_closed(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project entry_closed: settle the entry status."""

    entry.status = "closed"


def _apply_entry_quarantined(entry: LedgerEntry, event: LedgerEvent) -> None:
    """Project entry_quarantined: settle the status with the disclosed reason."""

    entry.status = "quarantined"
    entry.quarantine_reason = event.payload.get("reason")


_ENTRY_EVENT_APPLIERS: dict[str, Callable[[LedgerEntry, LedgerEvent], None]] = {
    "entry_opened": _apply_entry_opened,
    "hypothesis_set": _apply_hypothesis_set,
    "metric_chosen": _apply_metric_chosen,
    "material_step_started": _apply_step_started,
    "material_step_finalized": _apply_step_finalized,
    "observation_recorded": _apply_observation,
    "note": _apply_note,
    "verdict_set": _apply_verdict_set,
    "entry_closed": _apply_entry_closed,
    "entry_quarantined": _apply_entry_quarantined,
}


def _project_entries(events: list[LedgerEvent]) -> dict[str, LedgerEntry]:
    """Fold the event stream into per-entry projections (pure, derived)."""

    entries: dict[str, LedgerEntry] = {}
    for event in events:
        entry = entries.setdefault(event.entry_id, LedgerEntry(entry_id=event.entry_id))
        applier = _ENTRY_EVENT_APPLIERS.get(event.kind)
        if applier is not None:
            applier(entry, event)
    return entries


_ACTIVE: ContextVar[ExperimentLedger | None] = ContextVar(
    "torchlens_experiment_ledger", default=None
)


def active_ledger() -> ExperimentLedger | None:
    """The ledger armed in THIS context (ContextVar-scoped), or None."""

    return _ACTIVE.get()


class ExperimentLedger:
    """One armed, file-backed, single-writer experiment ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        hypothesis: str | None = None,
        metric: str | None = None,
        question: str | None = None,
        actor: str = "user",
        on_record_error: Literal["quarantine", "raise"] = "quarantine",
    ) -> None:
        if on_record_error not in ("quarantine", "raise"):
            raise BundleExperimentError(
                f"on_record_error must be 'quarantine' or 'raise', got {on_record_error!r}",
                code="ledger_option_invalid",
            )
        self.path = Path(path)
        self.actor = str(actor)
        self.on_record_error: str = on_record_error
        self.ledger_id = uuid.uuid4().hex[:16]
        self.writer_id = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._events: list[LedgerEvent] = []
        self._entry_counter = 0
        self._current_entry_id: str | None = None
        self._quarantined_entries: set[str] = set()
        self._handle: Any = None
        self._lock_handle: Any = None
        self._token: Any = None
        self._warned_once = False
        # ---- arm-time preflight: FAIL CLOSED before any material work ----
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # The LOCK lives on a sidecar file recovery never replaces
            # (compaction os.replace()s the stream inode, which would orphan
            # a lock held on it).
            self._lock_handle = self.path.with_name(self.path.name + ".lock").open(
                "a", encoding="utf-8"
            )
            self._lock_file()
            exists = self.path.exists()
            if exists:
                recovered, torn = _read_stream(self.path)
                self._events = recovered
                self._entry_counter = len({event.entry_id for event in recovered})
                if torn:
                    warnings.warn(
                        TorchLensWarning(
                            f"experiment ledger {self.path} carried one torn "
                            "tail line (an interrupted write); it was "
                            "discarded and is disclosed here. All "
                            f"{len(recovered)} finalized events recovered.",
                            code="ledger_torn_tail_discarded",
                        ),
                        stacklevel=3,
                    )
                    # Compact (drop the torn tail) atomically; safe under the
                    # sidecar lock.
                    _rewrite_stream(self.path, recovered)
            self._handle = self.path.open("a", encoding="utf-8")
            if not exists:
                manifest = {
                    "schema": LEDGER_SCHEMA,
                    "ledger_id": self.ledger_id,
                    "writer_id": self.writer_id,
                    "on_record_error": self.on_record_error,
                }
                self._handle.write(_canonical(manifest) + "\n")
                self._handle.flush()
                os.fsync(self._handle.fileno())
        except BundleExperimentError:
            raise
        except OSError as exc:
            raise BundleExperimentError(
                f"experiment ledger could not arm at {self.path}: "
                f"{type(exc).__name__}: {exc}. Arm-time preflight fails "
                "closed BEFORE any material work (a dead sink must never eat "
                "an experiment's claims).",
                code="ledger_arm_failed",
                path=str(self.path),
            ) from exc
        # Arm the context and open entry 1 (visibly DRAFT without hypothesis).
        current = _ACTIVE.get()
        if current is not None and current is not self:
            self._unlock_and_close()
            raise BundleExperimentError(
                "another experiment ledger is already armed in this context; "
                "nested different-scope arming refuses (same-scope re-entry "
                "is idempotent). Close the active ledger first.",
                code="ledger_already_armed",
                active_path=str(current.path),
            )
        self._token = _ACTIVE.set(self)
        self.entry(question=question, hypothesis=hypothesis, metric=metric)

    # ---- file plumbing ----------------------------------------------------

    def _lock_file(self) -> None:
        """Take the exclusive sidecar flock or refuse typed (v1 is single-writer)."""

        import fcntl

        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_handle.close()
            raise BundleExperimentError(
                f"experiment ledger {self.path} is locked by another writer; "
                "v1 is single-writer (competing writers refuse, a future "
                "merge is never last-write-wins).",
                code="ledger_locked",
                path=str(self.path),
            ) from exc

    def _unlock_and_close(self) -> None:
        """Best-effort close of the stream and sidecar-lock handles (disarm path)."""

        if self._handle is not None:
            with contextlib.suppress(OSError, ValueError):
                self._handle.close()
            self._handle = None
        lock_handle = getattr(self, "_lock_handle", None)
        if lock_handle is not None:
            with contextlib.suppress(OSError, ValueError):
                lock_handle.close()
            self._lock_handle = None

    # ---- event writing ----------------------------------------------------

    def _append(self, entry_id: str, kind: str, payload: dict[str, Any]) -> LedgerEvent:
        """Append + fsync ONE event (durable before it is reported)."""

        previous = self._events[-1] if self._events else None
        event = LedgerEvent(
            seq=(previous.seq + 1) if previous is not None else 1,
            event_id=uuid.uuid4().hex[:16],
            entry_id=entry_id,
            kind=kind,
            actor=self.actor,
            payload=payload,
            prev_digest=previous.event_digest if previous is not None else None,
        )
        if self._handle is None or self._handle.closed:
            raise OSError("ledger sink is closed")
        self._handle.write(event.to_line() + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._events.append(event)
        return event

    def _record(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        material_result: Any = None,
        material_action_completed: bool = False,
    ) -> Any:
        """The D4g failure state machine around ONE semantic event."""

        entry_id = self._require_entry()
        if entry_id in self._quarantined_entries:
            # A quarantined entry's rows stop being claims; drop silently
            # into the void is banned, so this is a typed refusal for
            # explicit calls (material emitters check first).
            raise BundleExperimentError(
                f"ledger entry {entry_id} is quarantined; its rows stopped "
                "being claims after a write failure. Open a new entry.",
                code="ledger_entry_quarantined",
                entry_id=entry_id,
            )
        try:
            self._append(entry_id, kind, payload)
            return material_result
        except (OSError, ValueError) as exc:
            if not material_action_completed or self.on_record_error == "raise":
                raise BundleExperimentError(
                    f"experiment ledger write failed ({type(exc).__name__}: {exc}); "
                    f"material_action_completed={material_action_completed}. "
                    "The material result (if any) rides fields['result'].",
                    code="ledger_write_failed",
                    material_action_completed=material_action_completed,
                    result=material_result,
                    entry_id=entry_id,
                ) from exc
            self._quarantine(entry_id, reason=f"{type(exc).__name__}: {exc}")
            return material_result

    def _quarantine(self, entry_id: str, *, reason: str) -> None:
        """Warn once, quarantine + disarm the entry, best-effort event."""

        self._quarantined_entries.add(entry_id)
        if not self._warned_once:
            self._warned_once = True
            last_seq = self._events[-1].seq if self._events else 0
            warnings.warn(
                TorchLensWarning(
                    f"experiment ledger write failed AFTER material work "
                    f"(entry {entry_id}, last durable seq {last_seq}, path "
                    f"{self.path}): the entry is QUARANTINED (its rows stop "
                    "being claims) and the material result is returned "
                    f"unharmed. Reason: {reason}. material_action_completed=True",
                    code="ledger_entry_quarantined",
                ),
                stacklevel=4,
            )
        with contextlib.suppress(OSError, ValueError):
            self._append(entry_id, "entry_quarantined", {"reason": reason})

    def _require_entry(self) -> str:
        """Return the open entry id or refuse typed (semantic writes need an entry)."""

        if self._current_entry_id is None:
            raise BundleExperimentError(
                "no open ledger entry; arm with ledger(path, ...) or open one with .entry()",
                code="ledger_no_entry",
            )
        return self._current_entry_id

    # ---- public semantic surface (INTERPRETIVE fields, actor-stamped) ----

    def entry(
        self,
        *,
        question: str | None = None,
        hypothesis: str | None = None,
        metric: str | None = None,
    ) -> ExperimentLedger:
        """Open the NEXT entry (one semantic question per entry, D4b)."""

        self._entry_counter += 1
        entry_id = f"e{self._entry_counter}"
        self._current_entry_id = entry_id
        self._record("entry_opened", {"question": question})
        if hypothesis is not None:
            self._record("hypothesis_set", {"hypothesis": hypothesis})
        if metric is not None:
            self._record("metric_chosen", {"metric": metric})
        return self

    def hypothesis(self, text: str) -> ExperimentLedger:
        """Declare (or amend) the entry's hypothesis; declaration seq is fact."""

        self._record("hypothesis_set", {"hypothesis": str(text)})
        return self

    def metric(self, text: str) -> ExperimentLedger:
        """Declare (or amend) the entry's metric CHOICE; declaration seq is fact."""

        self._record("metric_chosen", {"metric": str(text)})
        return self

    def note(self, text: str) -> ExperimentLedger:
        """Record one free-text note event on the open entry."""

        self._record("note", {"text": str(text)})
        return self

    def observe(self, text: str, *, value: Any = None) -> ExperimentLedger:
        """Record an EXPLICIT observation (reads are quiet; recording is an act)."""

        payload: dict[str, Any] = {"text": str(text)}
        if value is not None:
            payload["value"] = float(value) if isinstance(value, (int, float)) else str(value)
        self._record("observation_recorded", payload)
        return self

    def verdict(
        self,
        token: str,
        *,
        text: str | None = None,
        basis: tuple[str, ...] | list[str] = (),
    ) -> ExperimentLedger:
        """Set the entry verdict (closed tokens; free text beside).

        An empty ``basis`` is legal and renders "asserted; no recorded
        basis". TorchLens LINTS a verdict over zero recorded observations
        (one typed disclosure event) and never blocks or computes one.
        """

        if token not in VERDICT_TOKENS:
            raise BundleExperimentError(
                f"verdict token {token!r} is outside the closed vocabulary "
                f"{sorted(VERDICT_TOKENS)} (free text goes in text=, the "
                "vocabulary never grows per call site)",
                code="ledger_verdict_invalid",
                token=token,
            )
        entry = self.entries().get(self._require_entry())
        previous = entry.verdict if entry is not None else None
        if entry is not None and not entry.observations and token != "not_assessed":
            self._record(
                "lint",
                {"text": "verdict set over zero recorded observations (disclosure, never a block)"},
            )
        self._record(
            "verdict_set",
            {
                "token": token,
                "text": text,
                "basis": [str(item) for item in basis],
                "amends": previous["token"] if previous else None,
            },
        )
        return self

    def close_entry(self, *, exploratory: bool = False) -> ExperimentLedger:
        """Close the current entry; a verdict (or exploratory flag) is required."""

        entry = self.entries().get(self._require_entry())
        if entry is not None and entry.verdict is None and not exploratory:
            raise BundleExperimentError(
                "closing an entry requires an explicit verdict (not_assessed "
                "counts) or exploratory=True — abandonment is entry STATUS, "
                "never a verdict, so a dropped experiment can never read as "
                "concluded.",
                code="ledger_close_without_verdict",
                entry_id=self._require_entry(),
            )
        self._record("entry_closed", {"exploratory": exploratory})
        return self

    # ---- material emission (MECHANICAL fields; used by verbs and 9b) ------

    def material_step(
        self,
        operation: str,
        *,
        refs: list[EvidenceRef] | None = None,
    ) -> str:
        """Write the durable started event BEFORE material work (D4g rule 2)."""

        step_id = uuid.uuid4().hex[:12]
        self._record(
            "material_step_started",
            {
                "step_id": step_id,
                "operation": operation,
                "refs": [ref.to_payload() for ref in refs or []],
            },
        )
        return step_id

    def finalize_step(
        self,
        step_id: str,
        *,
        outcome: str,
        refs: list[EvidenceRef] | None = None,
        material_result: Any = None,
    ) -> Any:
        """Finalize one step AFTER material work (quarantine on sink death)."""

        return self._record(
            "material_step_finalized",
            {
                "step_id": step_id,
                "outcome": outcome,
                "refs": [ref.to_payload() for ref in refs or []],
            },
            material_result=material_result,
            material_action_completed=True,
        )

    def relink(self, ref: EvidenceRef, *, reason: str) -> ExperimentLedger:
        """Explicit append-only administrative re-link (never silent)."""

        self._record("evidence_relinked", {"ref": ref.to_payload(), "reason": str(reason)})
        return self

    # ---- derived reads ------------------------------------------------------

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        """Every finalized event in seq order (a fresh immutable snapshot)."""

        return tuple(self._events)

    def entries(self) -> dict[str, LedgerEntry]:
        """Project the event stream into per-entry DERIVED views."""

        return _project_entries(self._events)

    # ---- lifecycle ----------------------------------------------------------

    def save(self) -> None:
        """Explicit durability check — RAISES on a dead sink (D4g rule 4)."""

        if self._handle is None or self._handle.closed:
            raise BundleExperimentError(
                "experiment ledger sink is closed", code="ledger_write_failed"
            )
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except (OSError, ValueError) as exc:
            raise BundleExperimentError(
                f"experiment ledger save() failed: {type(exc).__name__}: {exc}",
                code="ledger_write_failed",
            ) from exc

    def close(self) -> None:
        """Disarm this context and close the sink.

        Explicit close RAISES on a dead sink (D4g rule 4) — but the context
        is ALWAYS disarmed and the lock released first, so a failed close
        never leaves a zombie arm blocking the next experiment.
        """

        try:
            self.save()
        finally:
            if self._token is not None:
                with contextlib.suppress(ValueError):
                    _ACTIVE.reset(self._token)
                self._token = None
            if _ACTIVE.get() is self:
                _ACTIVE.set(None)
            self._unlock_and_close()

    def __enter__(self) -> ExperimentLedger:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def ledger(
    path: str | Path,
    *,
    hypothesis: str | None = None,
    metric: str | None = None,
    question: str | None = None,
    actor: str = "user",
    on_record_error: Literal["quarantine", "raise"] = "quarantine",
) -> ExperimentLedger:
    """Arm a file-backed experiment ledger in THIS context (notebook-first).

    Arms immediately and opens entry 1 (visibly DRAFT without a hypothesis);
    the with-form is sugar (context managers do not survive notebook cells).
    """

    return ExperimentLedger(
        path,
        hypothesis=hypothesis,
        metric=metric,
        question=question,
        actor=actor,
        on_record_error=on_record_error,
    )


# ---- artifact reading (shared by recovery and the MCP tools) --------------


def _read_stream(path: Path) -> tuple[list[LedgerEvent], bool]:
    """Recover finalized events; (events, torn_tail_discarded).

    A torn TAIL line (interrupted final write) is discarded and disclosed by
    the caller; a torn or tampered INTERIOR line refuses at the break — an
    interior gap means rows after it are unattested.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise BundleExperimentError(
            f"experiment ledger {path} is empty (no manifest line)",
            code="ledger_artifact_invalid",
            path=str(path),
        )
    manifest_line, *event_lines = lines
    try:
        manifest = loads_bounded(manifest_line)
        if manifest.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"schema {manifest.get('schema')!r} != {LEDGER_SCHEMA!r}")
    except (json.JSONDecodeError, ValueError) as exc:
        raise BundleExperimentError(
            f"experiment ledger {path} manifest is invalid: {exc}",
            code="ledger_artifact_invalid",
            path=str(path),
        ) from exc
    events: list[LedgerEvent] = []
    torn = False
    for index, line in enumerate(event_lines):
        is_tail = index == len(event_lines) - 1
        try:
            record = loads_bounded(line)
            event = LedgerEvent.from_line_payload(record)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if is_tail:
                torn = True
                break
            raise BundleExperimentError(
                f"experiment ledger {path} breaks at interior event line "
                f"{index + 2}: {exc}. Rows after an interior break are "
                "unattested; the load refuses at the break.",
                code="ledger_artifact_invalid",
                path=str(path),
                line=index + 2,
            ) from exc
        expected_prev = events[-1].event_digest if events else None
        expected_seq = (events[-1].seq + 1) if events else 1
        if event.prev_digest != expected_prev or event.seq != expected_seq:
            raise BundleExperimentError(
                f"experiment ledger {path} hash chain breaks at seq "
                f"{event.seq}: dropped, reordered, or edited events.",
                code="ledger_artifact_invalid",
                path=str(path),
                seq=event.seq,
            )
        events.append(event)
    return events, torn


def _rewrite_stream(path: Path, events: list[LedgerEvent]) -> None:
    """Atomically rewrite the artifact (recovery compaction only)."""

    manifest_line = path.read_text(encoding="utf-8").split("\n", 1)[0]
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(manifest_line + "\n")
        for event in events:
            handle.write(event.to_line() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_ledger_artifact(
    path: str | Path,
) -> tuple[list[LedgerEvent], dict[str, LedgerEntry], bool]:
    """Read-only artifact load for the serving tools (fresh mid-experiment).

    Returns (events, entry projections, torn_tail_discarded). Never locks,
    never writes: durability is per-event, so the on-disk artifact is fresh
    while the experiment is still running.
    """

    resolved = Path(path)
    if not resolved.exists():
        raise BundleExperimentError(
            f"no experiment ledger at {resolved}",
            code="ledger_artifact_invalid",
            path=str(resolved),
        )
    events, torn = _read_stream(resolved)
    return events, _project_entries(events), torn
