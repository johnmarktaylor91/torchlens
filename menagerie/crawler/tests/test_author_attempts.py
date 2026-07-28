"""Attempt records: identity, supersession, quarantine, and the feedback channel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.author_attempts import (
    ATTEMPT_RECORD_VERSION,
    AttemptRecordError,
    latest_attempt,
    list_attempts,
    new_attempt,
    prior_attempts_summary,
)


def _open(root: Path, **overrides: object):
    """Open one attempt with test defaults."""

    kwargs: dict = {
        "stable_id": "m1",
        "campaign_id": "c1-mech",
        "kind": "source-request",
    }
    kwargs.update(overrides)
    return new_attempt(root, **kwargs)


def test_new_attempt_creates_durable_record_and_dirs(tmp_path: Path) -> None:
    """A fresh attempt persists its record, scratch, broker, and event log."""

    handle = _open(tmp_path)
    assert handle.number == 1
    assert handle.status == "created"
    assert handle.paths.scratch.is_dir()
    assert handle.paths.broker.is_dir()
    record = json.loads(handle.paths.record.read_text(encoding="utf-8"))
    assert record["record_version"] == ATTEMPT_RECORD_VERSION
    assert record["attempt_nonce"] == handle.nonce
    events = handle.paths.events.read_text(encoding="utf-8").splitlines()
    assert json.loads(events[0])["event"] == "created"


def test_attempt_numbers_are_monotonic_and_dirs_never_shared(tmp_path: Path) -> None:
    """Attempts never share a directory; numbers strictly increase."""

    first = _open(tmp_path)
    second = _open(tmp_path)
    third = _open(tmp_path)
    assert [first.number, second.number, third.number] == [1, 2, 3]
    dirs = {first.paths.directory, second.paths.directory, third.paths.directory}
    assert len(dirs) == 3


def test_new_attempt_supersedes_open_priors_and_quarantines_stray_results(
    tmp_path: Path,
) -> None:
    """Opening attempt N+1 terminalizes open priors and quarantines their results.

    This is the structural half of the anti-laundering discipline: a late
    completion from a superseded attempt lands in a directory that is never
    read as a result.
    """

    first = _open(tmp_path)
    first.update(status="stage2-running")
    # A still-running orphan wrote a late result into the old attempt.
    late = first.paths.directory / "result.json"
    late.write_text('{"stale": true}', encoding="utf-8")

    second = _open(tmp_path)
    reloaded = list_attempts(tmp_path)[0]
    assert reloaded.status == "superseded"
    assert reloaded.record["superseded"]["by_nonce"] == second.nonce
    assert not late.exists()
    quarantined = list(reloaded.paths.quarantine.iterdir())
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text(encoding="utf-8")) == {"stale": True}
    assert reloaded.record["quarantine"][0]["reason"] == "superseded:retry"


def test_completed_attempts_are_not_superseded(tmp_path: Path) -> None:
    """A completed attempt is history, not supersedable state."""

    first = _open(tmp_path)
    first.update(status="completed")
    _open(tmp_path)
    assert list_attempts(tmp_path)[0].status == "completed"


def test_terminal_status_refuses_transitions(tmp_path: Path) -> None:
    """Terminal records refuse further lifecycle transitions."""

    handle = _open(tmp_path)
    handle.update(status="failed")
    with pytest.raises(AttemptRecordError):
        handle.update(status="stage1-running")


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    """The status vocabulary is closed."""

    handle = _open(tmp_path)
    with pytest.raises(AttemptRecordError):
        handle.update(status="wandering")


def test_latest_attempt_filters_by_status_and_stable_id(tmp_path: Path) -> None:
    """Resume lookups match on model identity and lifecycle status."""

    first = _open(tmp_path)
    first.update(status="sources-published")
    second = _open(tmp_path, kind="author")
    second.update(status="stage2-running")

    found = latest_attempt(
        tmp_path, stable_id="m1", statuses=frozenset({"stage2-running"})
    )
    assert found is not None and found.nonce == second.nonce
    assert latest_attempt(tmp_path, stable_id="other") is None
    # The first attempt was superseded by the second, so it no longer matches.
    assert latest_attempt(tmp_path, statuses=frozenset({"sources-published"})) is None


def test_prior_attempts_summary_carries_failure_reasons(tmp_path: Path) -> None:
    """The feedback channel names what went wrong, verbatim, per attempt."""

    first = _open(tmp_path)
    first.update(
        status="failed",
        outcome={
            "kind": "failure",
            "failure_stage": "stage1",
            "failure_reason": "primary-implementation-unfetchable",
        },
    )
    _open(tmp_path)
    summaries = prior_attempts_summary(tmp_path)
    assert summaries[0]["failure_reason"] == "primary-implementation-unfetchable"
    assert summaries[0]["attempt_number"] == 1


def test_unreadable_attempt_dirs_are_inert(tmp_path: Path) -> None:
    """A directory without a readable record cannot be resumed or published."""

    handle = _open(tmp_path)
    handle.paths.record.write_text("not json", encoding="utf-8")
    assert list_attempts(tmp_path) == []


def test_empty_stable_id_is_rejected(tmp_path: Path) -> None:
    """Attempts must bind a model identity."""

    with pytest.raises(AttemptRecordError):
        new_attempt(tmp_path, stable_id="  ", campaign_id=None, kind="author")
