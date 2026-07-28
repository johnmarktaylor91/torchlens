"""Durable per-attempt records for the headless author executor.

One **attempt** is one executor pass over one model's author lifecycle: stage 1
discovery, the machine-owned source broker, and stage 2 authoring, all inside a
single per-attempt directory. Nothing is ever shared between attempts, so a
superseded attempt structurally *cannot* publish into a live one — the executor
publishes only from the attempt it opened, and any stray output found inside a
superseded attempt directory is quarantined, never read as a result.

Layout, under the lane's author root (``<work>/<stable_id>/author``)::

    attempts/attempt-<NNN>-<nonce>/
        record.json        current durable state, rewritten atomically
        events.jsonl       append-only event log
        scratch/           cwd for the attempt's ``claude -p`` sessions
        broker/            resolver receipts, raw API evidence, probe outcomes
        quarantine/        late or superseded outputs, moved, never deleted

The record is the resume authority: a killed executor leaves the record behind,
and the next invocation reads it to decide whether to resume the recorded
provider session, replay a completed phase from its recorded output, or open
attempt N+1 with this attempt in ``prior_attempts``.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from menagerie.crawler.identity import fsync_directory, hash_bytes, utc_now
from menagerie.crawler.models import JsonObject

ATTEMPT_RECORD_VERSION = "menagerie.crawler.author-attempt.v1"
ATTEMPTS_DIR_NAME = "attempts"

#: Statuses from which an attempt can never progress again.
TERMINAL_STATUSES = frozenset({"completed", "failed", "superseded"})
#: Every status the record may carry, in rough lifecycle order.
KNOWN_STATUSES = frozenset(
    {
        "created",
        "stage1-running",
        "stage1-complete",
        "sources-published",
        "stage2-running",
        "supplement-running",
        "completed",
        "failed",
        "superseded",
    }
)

_ATTEMPT_DIR_PATTERN = re.compile(r"^attempt-(\d{3,})-([0-9a-f]{8,64})$")


class AttemptRecordError(ValueError):
    """Raised when an attempt record or directory is not exact."""


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one JSON object with a directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> Optional[JsonObject]:
    """Read one JSON object, returning ``None`` for absent or unreadable files."""

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def attempts_root(author_root: Union[str, Path]) -> Path:
    """Return the attempts registry directory under one author root.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory (the request envelope's parent).

    Returns
    -------
    pathlib.Path
        Registry directory holding every attempt of this model.
    """

    return Path(author_root) / ATTEMPTS_DIR_NAME


@dataclass(frozen=True)
class AttemptPaths:
    """Absolute paths owned by exactly one attempt."""

    directory: Path
    record: Path
    events: Path
    scratch: Path
    broker: Path
    quarantine: Path

    @classmethod
    def for_directory(cls, directory: Path) -> "AttemptPaths":
        """Derive every attempt-owned path from the attempt directory."""

        directory = Path(directory)
        return cls(
            directory=directory,
            record=directory / "record.json",
            events=directory / "events.jsonl",
            scratch=directory / "scratch",
            broker=directory / "broker",
            quarantine=directory / "quarantine",
        )


class AttemptHandle:
    """One live attempt: its paths, its durable record, and its event log."""

    def __init__(self, paths: AttemptPaths, record: JsonObject, *, clock: Callable[[], str]):
        self.paths = paths
        self.record = record
        self._clock = clock

    @property
    def nonce(self) -> str:
        """Return the attempt nonce."""

        return str(self.record["attempt_nonce"])

    @property
    def number(self) -> int:
        """Return the attempt number."""

        return int(self.record["attempt_number"])

    @property
    def status(self) -> str:
        """Return the current lifecycle status."""

        return str(self.record["status"])

    def update(self, **fields: Any) -> None:
        """Merge fields into the record and persist it atomically.

        Parameters
        ----------
        **fields:
            Top-level record fields to set. ``status`` is validated against the
            closed vocabulary; a terminal record refuses further transitions.

        Raises
        ------
        AttemptRecordError
            On an unknown status or a transition out of a terminal status.
        """

        new_status = fields.get("status")
        if new_status is not None:
            if new_status not in KNOWN_STATUSES:
                raise AttemptRecordError(f"unknown attempt status {new_status!r}")
            if self.status in TERMINAL_STATUSES and new_status != self.status:
                raise AttemptRecordError(
                    f"attempt {self.nonce} is terminal ({self.status}); "
                    f"refusing transition to {new_status}"
                )
        self.record.update(fields)
        self.record["updated_at"] = self._clock()
        _write_json_atomic(self.paths.record, self.record)

    def event(self, name: str, **fields: Any) -> None:
        """Append one event line to the attempt's durable event log."""

        line = json.dumps(
            {"at": self._clock(), "event": name, **fields},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with self.paths.events.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def quarantine(self, path: Path, *, reason: str) -> Optional[Path]:
        """Move one stray output into this attempt's quarantine directory.

        Parameters
        ----------
        path:
            File to quarantine. Absent files are ignored (nothing to launder).
        reason:
            Closed short reason recorded with the move.

        Returns
        -------
        pathlib.Path | None
            Quarantined destination, or ``None`` when the file was absent.
        """

        path = Path(path)
        if not path.is_file():
            return None
        self.paths.quarantine.mkdir(parents=True, exist_ok=True)
        destination = self.paths.quarantine / f"{uuid.uuid4().hex[:8]}-{path.name}"
        os.replace(path, destination)
        entries = list(self.record.get("quarantine", []))
        entries.append(
            {
                "original": str(path),
                "quarantined_to": str(destination),
                "reason": reason,
                "sha256": hash_bytes(destination.read_bytes()),
                "at": self._clock(),
            }
        )
        self.update(quarantine=entries)
        self.event("quarantined", original=str(path), destination=str(destination), reason=reason)
        return destination

    def mark_superseded(self, *, by_nonce: str, reason: str) -> None:
        """Terminalize this attempt as superseded and quarantine stray results.

        Any ``result.json`` / ``source-targets.json`` a still-running orphaned
        provider session may yet write into this attempt lands in a directory
        that is never read as a result; files already present are moved to
        quarantine now. ``discovery.json`` stays: it is crash-recovery input,
        never a publishable result.

        Parameters
        ----------
        by_nonce:
            Nonce of the superseding attempt.
        reason:
            Closed short reason (``retry``, ``timeout``, ``crash-recovery``).
        """

        if self.status in TERMINAL_STATUSES and self.status != "superseded":
            # A completed/failed attempt is history, not supersedable state.
            return
        for name in ("result.json", "source-targets.json"):
            self.quarantine(self.paths.directory / name, reason=f"superseded:{reason}")
            self.quarantine(self.paths.scratch / name, reason=f"superseded:{reason}")
        if self.status != "superseded":
            self.update(
                status="superseded",
                superseded={"at": self._clock(), "by_nonce": by_nonce, "reason": reason},
            )
            self.event("superseded", by_nonce=by_nonce, reason=reason)


def new_attempt(
    author_root: Union[str, Path],
    *,
    stable_id: str,
    campaign_id: Optional[str],
    kind: str,
    clock: Callable[[], str] = utc_now,
    nonce_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    inherit_from: Optional[Mapping[str, Any]] = None,
) -> AttemptHandle:
    """Open attempt N+1, superseding every still-open prior attempt.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    stable_id:
        Model identity the attempt serves.
    campaign_id:
        Campaign the attempt runs under, when known.
    kind:
        Request kind that opened the attempt (``source-request`` or ``author``).
    clock, nonce_factory:
        Injectable time and identity sources.
    inherit_from:
        Optional prior attempt record whose stage-1 session identity and
        discovery output this attempt may reuse (crash recovery).

    Returns
    -------
    AttemptHandle
        Live handle with a persisted ``created`` record.

    Raises
    ------
    AttemptRecordError
        When the stable id is empty.
    """

    if not stable_id.strip():
        raise AttemptRecordError("attempt stable_id must be non-empty")
    root = attempts_root(author_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = list_attempts(author_root, clock=clock)
    number = 1 + max((handle.number for handle in existing), default=0)
    nonce = nonce_factory()
    directory = root / f"attempt-{number:03d}-{nonce}"
    paths = AttemptPaths.for_directory(directory)
    paths.scratch.mkdir(parents=True, exist_ok=True)
    paths.broker.mkdir(parents=True, exist_ok=True)
    record: JsonObject = {
        "record_version": ATTEMPT_RECORD_VERSION,
        "stable_id": stable_id,
        "campaign_id": campaign_id,
        "kind": kind,
        "attempt_number": number,
        "attempt_nonce": nonce,
        "attempt_dir": str(directory),
        "created_at": clock(),
        "updated_at": clock(),
        "status": "created",
        "stage1": None,
        "stage2": None,
        "supplement": None,
        "broker": None,
        "outcome": None,
        "published": None,
        "superseded": None,
        "quarantine": [],
        "inherited": (
            {
                "attempt_nonce": inherit_from.get("attempt_nonce"),
                "attempt_number": inherit_from.get("attempt_number"),
                "stage1": inherit_from.get("stage1"),
                "discovery_path": inherit_from.get("discovery_path"),
            }
            if inherit_from is not None
            else None
        ),
    }
    handle = AttemptHandle(paths, record, clock=clock)
    _write_json_atomic(paths.record, record)
    handle.event("created", kind=kind, stable_id=stable_id)
    for prior in existing:
        if prior.status not in TERMINAL_STATUSES:
            prior.mark_superseded(by_nonce=nonce, reason="retry")
    return handle


def list_attempts(
    author_root: Union[str, Path],
    *,
    clock: Callable[[], str] = utc_now,
) -> list[AttemptHandle]:
    """Return every readable attempt under one author root, oldest first.

    Directories whose records are missing or unreadable are skipped: an attempt
    without a record cannot be resumed and cannot publish, so it is inert.
    """

    root = attempts_root(author_root)
    if not root.is_dir():
        return []
    handles: list[AttemptHandle] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or _ATTEMPT_DIR_PATTERN.match(entry.name) is None:
            continue
        paths = AttemptPaths.for_directory(entry)
        record = _read_json_object(paths.record)
        if record is None or record.get("record_version") != ATTEMPT_RECORD_VERSION:
            continue
        handles.append(AttemptHandle(paths, record, clock=clock))
    handles.sort(key=lambda handle: handle.number)
    return handles


def latest_attempt(
    author_root: Union[str, Path],
    *,
    stable_id: Optional[str] = None,
    statuses: Optional[frozenset[str]] = None,
    clock: Callable[[], str] = utc_now,
) -> Optional[AttemptHandle]:
    """Return the newest attempt matching the given filters.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    stable_id:
        Require this model identity when given.
    statuses:
        Restrict to these lifecycle statuses when given.
    clock:
        Injectable time source for the returned handle.
    """

    for handle in reversed(list_attempts(author_root, clock=clock)):
        if stable_id is not None and handle.record.get("stable_id") != stable_id:
            continue
        if statuses is not None and handle.status not in statuses:
            continue
        return handle
    return None


def prior_attempts_summary(
    author_root: Union[str, Path],
    *,
    limit: int = 5,
) -> list[JsonObject]:
    """Return the feedback channel: what earlier attempts did and why they ended.

    The newest ``limit`` non-live attempts are summarized for the envelope's
    ``prior_attempts`` block and the brief's "WHAT WENT WRONG LAST TIME"
    section. Verbatim failure reasons and checker findings ride along so a
    retry is never blind.

    Parameters
    ----------
    author_root:
        The lane's per-model author directory.
    limit:
        Maximum prior attempts to include, newest last.
    """

    summaries: list[JsonObject] = []
    for handle in list_attempts(author_root):
        record = handle.record
        if record.get("status") == "created":
            continue
        outcome = record.get("outcome") or {}
        published = record.get("published") or {}
        summaries.append(
            {
                "attempt_number": record.get("attempt_number"),
                "attempt_nonce": record.get("attempt_nonce"),
                "status": record.get("status"),
                "kind": record.get("kind"),
                "outcome": outcome,
                "failure_stage": outcome.get("failure_stage") if outcome else None,
                "failure_reason": outcome.get("failure_reason") if outcome else None,
                "checker_findings": record.get("checker_findings"),
                "prior_result_sha256": published.get("sha256") if published else None,
                "created_at": record.get("created_at"),
            }
        )
    return summaries[-limit:]
