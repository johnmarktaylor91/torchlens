"""Crash-safe, single-writer append-only JSONL ledger I/O."""

from __future__ import annotations

import errno
import fcntl
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import (
    ARTIFACT_EVENT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION_V3,
    GATE_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, payload_hash, stable_hash
from menagerie.crawler.models import AppendResult, JsonObject, TailRecoveryEvidence
from menagerie.crawler.schema import SCHEMA_FILES, PayloadValidationError, validate_payload

_SCHEMA_READ_VERSIONS: dict[str, frozenset[str]] = {
    MODEL_SCHEMA_VERSION: frozenset({MODEL_SCHEMA_VERSION}),
    MODEL_SCHEMA_VERSION_V3: frozenset({MODEL_SCHEMA_VERSION, MODEL_SCHEMA_VERSION_V3}),
    ATTEMPT_SCHEMA_VERSION: frozenset({ATTEMPT_SCHEMA_VERSION}),
    ATTEMPT_SCHEMA_VERSION_V3: frozenset({ATTEMPT_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION_V3}),
    GATE_SCHEMA_VERSION: frozenset({GATE_SCHEMA_VERSION}),
    GATE_SCHEMA_VERSION_V3: frozenset({GATE_SCHEMA_VERSION, GATE_SCHEMA_VERSION_V3}),
    ARTIFACT_EVENT_SCHEMA_VERSION: frozenset({ARTIFACT_EVENT_SCHEMA_VERSION}),
    OPERATIONAL_EVENT_SCHEMA_VERSION: frozenset({OPERATIONAL_EVENT_SCHEMA_VERSION}),
}

_MODEL_SCHEMA_VERSIONS = frozenset({MODEL_SCHEMA_VERSION, MODEL_SCHEMA_VERSION_V3})
_ATTEMPT_SCHEMA_VERSIONS = frozenset({ATTEMPT_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION_V3})
_GATE_SCHEMA_VERSIONS = frozenset({GATE_SCHEMA_VERSION, GATE_SCHEMA_VERSION_V3})


class LedgerError(RuntimeError):
    """Base class for canonical-ledger failures."""


class SingleWriterError(LedgerError):
    """Raised when another process owns the ledger writer lock."""


class LedgerCorruptionError(LedgerError):
    """Raised for a malformed complete line or invalid persisted record."""


class TornTailError(LedgerError):
    """Raised when a ledger ends in a non-newline-terminated tail."""


class LedgerConflictError(LedgerError):
    """Raised when one immutable identity maps to different payload bytes."""


class AttemptSlotResolutionError(LedgerError):
    """Raised when canonical history cannot satisfy one deterministic v3 slot."""


def _utc_now() -> str:
    """Return an RFC 3339 UTC timestamp.

    Returns
    -------
    str
        Current UTC timestamp ending in ``Z``.
    """

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    """Fsync a directory after creating or replacing one of its entries.

    Parameters
    ----------
    path:
        Directory to synchronize.
    """

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _logical_payload(record: Mapping[str, Any]) -> JsonObject:
    """Remove append-assigned fields for idempotency comparison.

    Parameters
    ----------
    record:
        Persisted or proposed ledger record.

    Returns
    -------
    dict[str, Any]
        Logical immutable payload.
    """

    omitted = {"payload_sha256", "ledger_seq", "record_revision", "record_seq"}
    return {key: deepcopy(value) for key, value in record.items() if key not in omitted}


def deterministic_attempt_id(
    *, work_id: str, execution_identity: str, cold_index: int, mode: str
) -> str:
    """Derive the immutable identity for one execution slot.

    Parameters
    ----------
    work_id, execution_identity:
        Exact work generation and executable dependency closure.
    cold_index:
        Zero-based cold-run index.
    mode:
        Exact isolated execution mode.

    Returns
    -------
    str
        Canonical attempt identity used by the ledger and worker request.

    Raises
    ------
    ValueError
        If any slot component is empty or the cold index is negative.
    """

    if not work_id.strip() or not execution_identity.strip() or not mode.strip():
        raise ValueError("attempt slot identities and mode must be non-empty")
    if cold_index < 0:
        raise ValueError("attempt slot cold_index must be non-negative")
    return stable_hash(
        {
            "work_id": work_id,
            "execution_identity": execution_identity,
            "cold_index": cold_index,
            "mode": mode,
        }
    )


def resolve_attempt_slot(
    records: Iterable[Mapping[str, Any]],
    *,
    work_id: str,
    execution_identity: str,
    cold_index: int,
    mode: str,
) -> Optional[JsonObject]:
    """Resolve one deterministic slot from canonical attempt history.

    The function intentionally performs no scratch lookup and never chooses a
    highest or latest record. A matching v2, duplicate, or slot-contradictory row
    is visible immutable history but cannot silently satisfy or rerun the slot.
    Raw-receipt and parent-proof authentication remains the reducer's authority
    boundary; the Phase-2 integrator supplies only authenticated current-v3 rows.

    Parameters
    ----------
    records:
        Canonical attempt rows, including immutable legacy history.
    work_id, execution_identity, cold_index, mode:
        Exact slot tuple used to derive the attempt identity.

    Returns
    -------
    dict[str, Any] | None
        Defensive copy of the sole exact v3 row, or ``None`` when absent.

    Raises
    ------
    AttemptSlotResolutionError
        If same-ID history is duplicate, legacy, malformed, or contradicts the
        exact slot tuple.
    """

    attempt_id = deterministic_attempt_id(
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=cold_index,
        mode=mode,
    )
    matches = [record for record in records if record.get("attempt_id") == attempt_id]
    if not matches:
        return None
    if len(matches) != 1:
        raise AttemptSlotResolutionError(
            f"deterministic attempt slot has duplicate canonical rows: {attempt_id}"
        )
    record = matches[0]
    if record.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
        raise AttemptSlotResolutionError(
            f"deterministic attempt slot is not current v3: {attempt_id}"
        )
    try:
        validate_payload(record, ATTEMPT_SCHEMA_VERSION_V3)
    except PayloadValidationError as exc:
        raise AttemptSlotResolutionError(
            f"deterministic attempt slot has an invalid v3 row: {attempt_id}"
        ) from exc
    identities = record.get("identities")
    retries = record.get("retries")
    invocation = record.get("invocation")
    raw_receipt = record.get("raw_award_receipt")
    observed_modes = {
        value
        for value in (
            record.get("mode"),
            invocation.get("mode") if isinstance(invocation, Mapping) else None,
            raw_receipt.get("requested_mode") if isinstance(raw_receipt, Mapping) else None,
        )
        if value is not None
    }
    if (
        record.get("work_id") != work_id
        or not isinstance(identities, Mapping)
        or identities.get("execution") != execution_identity
        or not isinstance(retries, Mapping)
        or retries.get("stage_attempt") != cold_index + 1
        or observed_modes - {mode}
    ):
        raise AttemptSlotResolutionError(
            f"deterministic attempt row contradicts its exact slot: {attempt_id}"
        )
    return deepcopy(dict(record))


def _identity_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the immutable replay/conflict key for a ledger record.

    Parameters
    ----------
    record:
        Ledger record.

    Returns
    -------
    tuple[Any, ...]
        Type-tagged identity key.
    """

    version = record.get("schema_version")
    if version in _MODEL_SCHEMA_VERSIONS:
        revision = record.get("record_revision")
        if revision is not None:
            return (version, record.get("stable_id"), revision)
        return (version, record.get("stable_id"), record.get("record_seq"))
    if version in _ATTEMPT_SCHEMA_VERSIONS:
        return (version, record.get("attempt_id"))
    if version in _GATE_SCHEMA_VERSIONS:
        return (version, record.get("gate_id"))
    if version == ARTIFACT_EVENT_SCHEMA_VERSION:
        return (version, record.get("artifact_event_id"))
    return (version, record.get("event_id") or record.get("proposal_id"))


def _verify_hash(record: Mapping[str, Any]) -> None:
    """Verify a persisted record's payload/revision digest.

    Parameters
    ----------
    record:
        Parsed persisted record.

    Raises
    ------
    LedgerCorruptionError
        If a present digest does not match canonical payload bytes.
    """

    digest_field = (
        "record_revision" if record.get("schema_version") in _MODEL_SCHEMA_VERSIONS else None
    )
    if digest_field is None and "payload_sha256" in record:
        digest_field = "payload_sha256"
    if digest_field is not None and record[digest_field] != payload_hash(record):
        raise LedgerCorruptionError(f"invalid {digest_field} for {_identity_key(record)!r}")


def scan_jsonl(path: Union[str, Path], *, validate: bool = True) -> list[JsonObject]:
    """Scan a complete JSONL ledger, stopping on corruption or a torn tail.

    Parameters
    ----------
    path:
        Ledger path. A missing file is an empty ledger.
    validate:
        Whether to schema-validate recognized crawler payloads.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records in byte order.

    Raises
    ------
    TornTailError
        If the final line lacks a newline.
    LedgerCorruptionError
        If any complete line is malformed, invalid, or hash-inconsistent.
    """

    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    records: list[JsonObject] = []
    with ledger_path.open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.endswith(b"\n"):
                raise TornTailError(
                    f"torn final line at byte {handle.tell() - len(line)} in {ledger_path}"
                )
            try:
                decoded = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerCorruptionError(
                    f"malformed complete JSON at {ledger_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(decoded, dict):
                raise LedgerCorruptionError(
                    f"non-object JSON record at {ledger_path}:{line_number}"
                )
            try:
                if validate and decoded.get("schema_version") in SCHEMA_FILES:
                    validate_payload(decoded)
                _verify_hash(decoded)
            except (PayloadValidationError, LedgerCorruptionError) as exc:
                raise LedgerCorruptionError(
                    f"invalid record at {ledger_path}:{line_number}: {exc}"
                ) from exc
            records.append(decoded)
    return records


def recover_torn_tail(
    path: Union[str, Path], *, evidence_path: Union[str, Path, None] = None
) -> Optional[TailRecoveryEvidence]:
    """Evidence and truncate only a non-newline-terminated final line.

    Complete malformed lines are never repaired. Recovery evidence is fsynced before
    the ledger is truncated.

    Parameters
    ----------
    path:
        Ledger to inspect.
    evidence_path:
        Optional append-only recovery ledger. Defaults beside the source ledger.

    Returns
    -------
    TailRecoveryEvidence | None
        Evidence for a removed tail, or None for an already complete ledger.

    Raises
    ------
    LedgerCorruptionError
        If a preceding complete line is malformed.
    """

    ledger_path = Path(path)
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return None
    data = ledger_path.read_bytes()
    if data.endswith(b"\n"):
        scan_jsonl(ledger_path)
        return None
    newline_offset = data.rfind(b"\n")
    good_end = newline_offset + 1
    prefix = data[:good_end]
    tail = data[good_end:]
    if prefix:
        temporary = ledger_path.with_name(f".{ledger_path.name}.recovery-scan")
        try:
            with temporary.open("wb") as handle:
                handle.write(prefix)
            scan_jsonl(temporary)
        finally:
            temporary.unlink(missing_ok=True)
    recovery_path = (
        Path(evidence_path)
        if evidence_path is not None
        else ledger_path.with_suffix(f"{ledger_path.suffix}.recovery.jsonl")
    )
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    tail_digest = hash_bytes(tail)
    recovered_at = _utc_now()
    evidence_record = {
        "ledger_path": str(ledger_path),
        "byte_offset": good_end,
        "byte_count": len(tail),
        "tail_sha256": tail_digest,
        "recovered_at": recovered_at,
    }
    with recovery_path.open("ab") as handle:
        handle.write(canonical_json_bytes(evidence_record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(recovery_path.parent)
    with ledger_path.open("r+b") as handle:
        handle.truncate(good_end)
        handle.flush()
        os.fsync(handle.fileno())
    evidence = TailRecoveryEvidence(
        ledger_path=ledger_path,
        byte_offset=good_end,
        byte_count=len(tail),
        tail_sha256=tail_digest,
        recovered_at=recovered_at,
        evidence_path=recovery_path,
    )
    return evidence


class JsonlLedger:
    """Single-writer append-only canonical ledger.

    Parameters
    ----------
    path:
        JSONL ledger path.
    schema_version:
        Current schema accepted for append. Its declared compatible immutable
        history versions are accepted on read.
    recover_tail:
        Whether startup may evidence and truncate a torn final line.
    """

    def __init__(
        self, path: Union[str, Path], schema_version: str, *, recover_tail: bool = True
    ) -> None:
        """Initialize and exclusively lock a canonical ledger.

        Parameters
        ----------
        path:
            JSONL ledger path.
        schema_version:
            Exact schema accepted by this ledger.
        recover_tail:
            Whether to recover a torn final line before scanning.
        """

        self.path = Path(path)
        self.schema_version = schema_version
        try:
            self.read_schema_versions = _SCHEMA_READ_VERSIONS[schema_version]
        except KeyError as exc:
            raise ValueError(
                f"schema is not a canonical ledger version: {schema_version!r}"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._lock_handle = self._lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SingleWriterError(f"ledger already has a writer: {self.path}") from exc
            raise
        self.recovery_evidence = recover_torn_tail(self.path) if recover_tail else None
        self._records = scan_jsonl(self.path)
        for record in self._records:
            if record.get("schema_version") not in self.read_schema_versions:
                self.close()
                raise LedgerCorruptionError(
                    f"schema {record.get('schema_version')!r} is not readable by "
                    f"current writer {self.schema_version!r} in {self.path}"
                )
        self._closed = False

    def __enter__(self) -> "JsonlLedger":
        """Return this locked writer.

        Returns
        -------
        JsonlLedger
            This writer.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release the exclusive writer lock.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    def __iter__(self) -> Iterator[JsonObject]:
        """Iterate over defensive copies of persisted records.

        Returns
        -------
        Iterator[dict[str, Any]]
            Records in ledger order.
        """

        return iter(deepcopy(self._records))

    def close(self) -> None:
        """Release the writer lock idempotently."""

        if getattr(self, "_closed", False):
            return
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._closed = True

    @property
    def records(self) -> Sequence[JsonObject]:
        """Return defensive copies of all records.

        Returns
        -------
        Sequence[dict[str, Any]]
            Persisted records in ledger order.
        """

        return tuple(deepcopy(self._records))

    def append(self, payload: Mapping[str, Any]) -> AppendResult:
        """Validate, idempotently append, flush, and fsync one complete line.

        Parameters
        ----------
        payload:
            Complete logical payload. Sequence and payload-hash fields may be omitted.

        Returns
        -------
        AppendResult
            Persisted record and whether a new line was written.

        Raises
        ------
        LedgerConflictError
            If an immutable identity already has a different logical payload.
        PayloadValidationError
            If the completed payload violates its schema.
        """

        if self._closed:
            raise LedgerError("cannot append through a closed ledger")
        record = deepcopy(dict(payload))
        if record.get("schema_version") != self.schema_version:
            raise PayloadValidationError(
                f"expected schema_version {self.schema_version!r}, "
                f"received {record.get('schema_version')!r}"
            )
        proposed_logical = _logical_payload(record)
        proposed_key = _identity_key(record)
        if all(value is not None for value in proposed_key[1:]):
            for existing in self._records:
                if _identity_key(existing) != proposed_key:
                    continue
                if _logical_payload(existing) == proposed_logical:
                    return AppendResult(deepcopy(existing), appended=False)
                raise LedgerConflictError(
                    f"conflicting replay for immutable identity {proposed_key!r}"
                )
        if self.schema_version in _MODEL_SCHEMA_VERSIONS:
            for existing in self._records:
                if (
                    existing["stable_id"] == record.get("stable_id")
                    and _logical_payload(existing) == proposed_logical
                ):
                    return AppendResult(deepcopy(existing), appended=False)
            expected_sequence = self._next_sequence("record_seq")
            if record.get("record_seq", expected_sequence) != expected_sequence:
                raise LedgerConflictError(
                    f"record_seq must be next local sequence {expected_sequence}"
                )
            record.setdefault("record_seq", expected_sequence)
            record["record_revision"] = payload_hash(record)
        else:
            expected_sequence = self._next_sequence("ledger_seq")
            if record.get("ledger_seq", expected_sequence) != expected_sequence:
                raise LedgerConflictError(
                    f"ledger_seq must be next local sequence {expected_sequence}"
                )
            record.setdefault("ledger_seq", expected_sequence)
            record["payload_sha256"] = payload_hash(record)
        key = _identity_key(record)
        logical = _logical_payload(record)
        for existing in self._records:
            if _identity_key(existing) != key:
                continue
            if _logical_payload(existing) == logical:
                return AppendResult(deepcopy(existing), appended=False)
            raise LedgerConflictError(f"conflicting replay for immutable identity {key!r}")
        validate_payload(record, self.schema_version)
        line = canonical_json_bytes(record) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)
        self._records.append(record)
        return AppendResult(deepcopy(record), appended=True)

    def _next_sequence(self, field: str) -> int:
        """Return the next monotonic local ledger sequence.

        Parameters
        ----------
        field:
            Sequence field for this schema.

        Returns
        -------
        int
            One greater than the maximum persisted value.
        """

        return max((int(record[field]) for record in self._records), default=0) + 1
