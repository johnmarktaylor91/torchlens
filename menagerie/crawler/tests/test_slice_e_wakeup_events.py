"""Slice E idempotent wakeup and operational-event vocabulary tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
    build_checkpoint_review_event,
    build_progress_notification_event,
    build_review_signoff_event,
)
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.schema import PayloadValidationError, validate_payload
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupBackend,
    WakeupManager,
    WakeupSpec,
    detect_wakeup_backend,
)

NOW = "2026-07-14T12:00:00Z"
RESET = "2026-07-14T13:00:00Z"


def _context() -> OperationalContext:
    """Return shared synthetic operational context.

    Returns
    -------
    OperationalContext
        Valid event context.
    """

    return OperationalContext("run-test", "machine-test", {"author": 2}, "env-test")


def test_pause_reset_creates_one_idempotent_wakeup_and_valid_events(tmp_path: Path) -> None:
    """Repeating one reset identity neither schedules nor appends twice."""

    ledger_path = tmp_path / "operational.jsonl"
    activated: list[WakeupSpec] = []
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = WakeupManager(
            tmp_path / "wakeups",
            ledger,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.LAUNCHD,
            activator=activated.append,
        )
        first = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="usage limit reached",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        replay = manager.record_pause_and_schedule(
            provider="anthropic",
            observed_response="usage limit reached",
            reset_at=RESET,
            context=_context(),
            created_at=NOW,
        )
        assert first.created
        assert not replay.created
        assert len(activated) == 1
    events = scan_jsonl(ledger_path)
    assert [event["event_kind"] for event in events] == ["usage-pause", "wakeup"]
    for event in events:
        validate_payload(event)


def test_wakeup_backend_feature_detection_prefers_ruled_platform_tools() -> None:
    """macOS uses launchd; other hosts prefer systemd and fall back to cron."""

    assert (
        detect_wakeup_backend(
            platform_name="Darwin", command_exists=lambda name: name == "launchctl"
        )
        is WakeupBackend.LAUNCHD
    )
    assert (
        detect_wakeup_backend(
            platform_name="Linux", command_exists=lambda name: name == "systemctl"
        )
        is WakeupBackend.SYSTEMD_TIMER
    )
    assert (
        detect_wakeup_backend(
            platform_name="FreeBSD", command_exists=lambda name: name == "crontab"
        )
        is WakeupBackend.CRON
    )


def test_review_and_progress_event_types_validate_strictly(tmp_path: Path) -> None:
    """All three additive event kinds validate and unknown kinds remain closed."""

    snapshot = FunnelSnapshot(runs=7, deferred=1, skipped=1, failed=1)
    logical_events = [
        build_checkpoint_review_event(
            models_completed=10,
            funnel_snapshot=snapshot,
            report_path="menagerie/crawler/views/review-10.json",
            context=_context(),
            created_at=NOW,
        ),
        build_review_signoff_event(
            approved_by_note="JMT approved review report 10",
            resume_after=10,
            context=_context(),
            created_at=NOW,
        ),
        build_progress_notification_event(
            models_completed=10,
            milestone=10,
            funnel_snapshot=snapshot,
            context=_context(),
            created_at=NOW,
        ),
    ]
    with JsonlLedger(tmp_path / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        persisted = [ledger.append(event).record for event in logical_events]
    assert {event["event_kind"] for event in persisted} == {
        "checkpoint-review",
        "review-signoff",
        "progress-notification",
    }
    for event in persisted:
        validate_payload(event)
    unknown = deepcopy(persisted[0])
    unknown["event_kind"] = "unknown-event"
    with pytest.raises(PayloadValidationError):
        validate_payload(unknown)
