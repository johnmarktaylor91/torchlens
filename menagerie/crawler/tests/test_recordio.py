"""Append-only JSONL durability, recovery, and idempotency tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION_V3
from menagerie.crawler.recordio import (
    AttemptSlotResolutionError,
    JsonlLedger,
    LedgerCorruptionError,
    LedgerConflictError,
    SingleWriterError,
    deterministic_attempt_id,
    recover_torn_tail,
    resolve_attempt_slot,
    scan_jsonl,
)
from menagerie.crawler.schema import PayloadValidationError
from menagerie.crawler.tests.conftest import make_attempt


def test_append_is_persisted_and_idempotent(tmp_path: Path) -> None:
    """A replay returns the original fsynced fact without another line.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    payload = make_attempt()
    payload.pop("ledger_seq")
    payload.pop("payload_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        first = ledger.append(payload)
        replay = ledger.append(payload)
        assert first.appended
        assert not replay.appended
        assert first.record == replay.record
    assert len(scan_jsonl(path)) == 1


def test_single_writer_lock_is_exclusive(tmp_path: Path) -> None:
    """Two live canonical writers cannot own one ledger.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3):
        with pytest.raises(SingleWriterError):
            JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)


def test_torn_tail_is_evidenced_before_truncation(tmp_path: Path) -> None:
    """Recovery retains offset/hash evidence and preserves valid facts.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(make_attempt())
    with path.open("ab") as handle:
        handle.write(b'{"schema_version":"menagerie.crawler.attempt.v2"')
    evidence = recover_torn_tail(path)
    assert evidence is not None
    assert evidence.byte_count > 0
    assert evidence.evidence_path.exists()
    assert len(scan_jsonl(path)) == 1


def test_complete_malformed_line_is_never_recovered(tmp_path: Path) -> None:
    """A newline-terminated malformed record is hard corruption.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b"not-json\n")
    with pytest.raises(LedgerCorruptionError):
        recover_torn_tail(path)


def test_v3_writer_reads_v2_history_but_appends_only_v3(tmp_path: Path) -> None:
    """Mixed immutable attempt history is readable without enabling legacy appends.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "mixed-attempts.jsonl"
    legacy = make_attempt()
    legacy["schema_version"] = ATTEMPT_SCHEMA_VERSION
    for field in (
        "capability_observation",
        "execution_read_manifest_identity",
        "raw_award_receipt",
        "raw_award_receipt_sha256",
        "parent_attestation",
        "unattested_partial",
    ):
        legacy.pop(field)
    legacy["worker_receipt"].pop("output_value_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION) as ledger:
        ledger.append(legacy)

    current = make_attempt()
    current["schema_version"] = ATTEMPT_SCHEMA_VERSION_V3
    current["attempt_id"] = "attempt-v3"
    current["execution_read_manifest_identity"] = "sha256:" + "e" * 64
    current["raw_award_receipt"] = {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": "nonce-v3",
        "request_sha256": "sha256:" + "1" * 64,
        "stable_id": current["stable_id"],
        "work_id": current["work_id"],
        "execution_identity": current["identities"]["execution"],
        "recipe_revision": current["identities"]["recipe"],
        "code_manifest_identity": "sha256:" + "2" * 64,
        "input_identity": "sha256:" + "3" * 64,
        "requested_mode": current["mode"],
        "observation": current["worker_receipt"],
    }
    current["raw_award_receipt_sha256"] = "sha256:" + "4" * 64
    current["parent_attestation"] = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": "nonce-v3",
        "request_sha256": "sha256:" + "1" * 64,
        "completion_line_sha256": "sha256:" + "5" * 64,
        "named_raw_award_receipt_sha256": "sha256:" + "4" * 64,
        "exit_code": 0,
        "signal": None,
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": 1,
        "stdout_sha256": "sha256:" + "6" * 64,
        "stderr_sha256": "sha256:" + "7" * 64,
        "started_at": current["started_at"],
        "finished_at": current["finished_at"],
        "attestation_sha256": "sha256:" + "8" * 64,
    }
    current["unattested_partial"] = None
    current.pop("ledger_seq")
    current.pop("payload_sha256")

    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert [row["schema_version"] for row in ledger.records] == [ATTEMPT_SCHEMA_VERSION]
        ledger.append(current)
        with pytest.raises(PayloadValidationError):
            ledger.append(legacy)
    assert [row["schema_version"] for row in scan_jsonl(path)] == [
        ATTEMPT_SCHEMA_VERSION,
        ATTEMPT_SCHEMA_VERSION_V3,
    ]


def test_resolve_attempt_slot_reuses_one_exact_current_row() -> None:
    """The canonical ledger row satisfies its deterministic slot without replay."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    attempt_id = deterministic_attempt_id(
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )
    attempt = make_attempt(attempt_id=attempt_id, execution_identity=execution_identity)

    resolved = resolve_attempt_slot(
        (attempt,),
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )

    assert resolved == attempt
    assert resolved is not attempt


@pytest.mark.parametrize("mutation", ["legacy", "mode", "duplicate"])
def test_resolve_attempt_slot_rejects_noncanonical_same_id(mutation: str) -> None:
    """Legacy, contradictory, and duplicate same-ID rows fail closed."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    attempt_id = deterministic_attempt_id(
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )
    attempt = make_attempt(attempt_id=attempt_id, execution_identity=execution_identity)
    records = [attempt]
    if mutation == "legacy":
        attempt["schema_version"] = ATTEMPT_SCHEMA_VERSION
    elif mutation == "mode":
        attempt["mode"] = "train"
    else:
        records.append(dict(attempt))

    with pytest.raises(AttemptSlotResolutionError):
        resolve_attempt_slot(
            records,
            work_id=work_id,
            execution_identity=execution_identity,
            cold_index=0,
            mode="eval",
        )


def test_attempt_timestamp_change_remains_an_immutable_conflict(tmp_path: Path) -> None:
    """M-02 retains timestamp fields in logical replay comparisons."""

    path = tmp_path / "attempts.jsonl"
    attempt = make_attempt()
    attempt.pop("ledger_seq")
    attempt.pop("payload_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(attempt)
        attempt["finished_at"] = "2026-07-16T12:00:01Z"
        with pytest.raises(LedgerConflictError):
            ledger.append(attempt)
