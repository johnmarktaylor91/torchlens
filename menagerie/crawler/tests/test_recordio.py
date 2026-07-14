"""Append-only JSONL durability, recovery, and idempotency tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerCorruptionError,
    SingleWriterError,
    recover_torn_tail,
    scan_jsonl,
)
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
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION) as ledger:
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
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION):
        with pytest.raises(SingleWriterError):
            JsonlLedger(path, ATTEMPT_SCHEMA_VERSION)


def test_torn_tail_is_evidenced_before_truncation(tmp_path: Path) -> None:
    """Recovery retains offset/hash evidence and preserves valid facts.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION) as ledger:
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
