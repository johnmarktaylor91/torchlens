"""Disposable SQLite rebuild tests."""

from __future__ import annotations

from pathlib import Path

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.state import load_current_records, rebuild_state
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_failed_attempt,
    make_model,
)


def test_sqlite_rebuild_materializes_current_revision(tmp_path: Path) -> None:
    """SQLite state is reproducibly rebuilt from intake and JSONL ledgers.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        persisted = reducer.append_model(first).record
        second = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
        second["record_seq"] = 2
        second["parent_revision"] = persisted["record_revision"]
        second["status"]["supersedes_revision"] = persisted["record_revision"]
        second["notes"] = "superseding evidence"
        reducer.append_model(second)
    database = tmp_path / "state.sqlite3"
    summary = rebuild_state(database, ["m_example"], paths)
    current = load_current_records(database)
    assert summary.model_revision_count == 2
    assert summary.current_count == 1
    assert current["m_example"]["notes"] == "superseding evidence"
