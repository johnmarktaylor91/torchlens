"""Slice E effort-cap and root-cause fingerprint tests."""

from __future__ import annotations

import pytest

from menagerie.crawler.effort import (
    CapFailureRecord,
    EffortCapExceeded,
    EffortGrant,
    EffortTracker,
    RepeatedRootCauseError,
    StageCap,
    _EFFORT_TRANSITIONS,
    _EffortMetric,
    _EffortTransition,
    fingerprint_root_cause,
)


def test_stage_cap_failure_is_typed_and_recorded_at_actual_stage() -> None:
    """A denied attempt records actual-stage cap exhaustion before raising."""

    recorded: list[CapFailureRecord] = []
    tracker = EffortTracker({"fetch": StageCap(attempts=1)}, recorder=recorded.append)
    tracker.consume("fetch", attempts=1)
    with pytest.raises(EffortCapExceeded) as caught:
        tracker.consume("fetch", attempts=1)
    assert caught.value.record.actual_stage == "fetch"
    assert caught.value.record.reason_code == "effort-cap-exhausted"
    assert recorded == [caught.value.record]


def test_identical_root_cause_stops_on_second_occurrence() -> None:
    """The same stage/fingerprint is recognized instead of blindly retried."""

    tracker = EffortTracker({"forward": StageCap(attempts=5)})
    fingerprint = fingerprint_root_cause(
        "forward", "exception", "RuntimeError", "same deterministic failure"
    )
    first = tracker.record_root_cause("forward", fingerprint)
    assert first.retry_allowed
    with pytest.raises(RepeatedRootCauseError) as caught:
        tracker.record_root_cause("forward", fingerprint)
    assert caught.value.record.metric == "root-cause-repeat"
    assert caught.value.record.root_cause_fingerprint == fingerprint


def test_explicit_grant_extends_only_its_stage_cap() -> None:
    """A unique append-only grant permits additional accounted work."""

    tracker = EffortTracker({"source": StageCap(bytes=10)})
    tracker.consume("source", bytes_used=10)
    tracker.add_grant(
        EffortGrant(
            grant_id="grant-1",
            stage="source",
            reason="human approved one bounded extension",
            granted_by="review-note-17",
            bytes=5,
        )
    )
    assert tracker.consume("source", bytes_used=5).bytes == 15


@pytest.mark.parametrize(
    ("increments", "metric"),
    [
        ({"attempts": 1, "seconds": 1.0, "bytes_used": 1}, "attempts"),
        ({"attempts": 0, "seconds": 1.0, "bytes_used": 1}, "seconds"),
        ({"attempts": 0, "seconds": 0.0, "bytes_used": 1}, "bytes"),
    ],
)
def test_effort_transition_table_is_exhaustive_and_preserves_diagnostic_order(
    increments: dict[str, int | float], metric: str
) -> None:
    """Every budget dimension is ruled and the first exceeded cap stays exact."""

    assert tuple(transition.metric for transition in _EFFORT_TRANSITIONS) == tuple(_EffortMetric)
    assert all(isinstance(value, _EffortTransition) for value in _EFFORT_TRANSITIONS)
    tracker = EffortTracker({"fetch": StageCap(attempts=0, seconds=0.0, bytes=0)})
    with pytest.raises(
        EffortCapExceeded,
        match=rf"fetch effort cap exhausted: {metric} 1 > 0",
    ):
        tracker.consume("fetch", **increments)
