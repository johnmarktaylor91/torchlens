"""Tests for the operator attempt-latency report built from wrapper telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from menagerie.crawler.tools.checker_latency import (
    build_report,
    format_report,
    iter_attempt_records,
    main,
    summarize,
)


def _attempt(
    *,
    attempt: int = 1,
    duration: float,
    budget: float = 600.0,
    timed_out: bool = False,
    classification: str = "success",
    gate_kind: str = "fidelity",
    item_count: int = 1,
) -> dict[str, Any]:
    """Build one attempt telemetry record.

    Parameters
    ----------
    attempt:
        Attempt ordinal.
    duration:
        Recorded wall seconds.
    budget:
        Effective per-attempt bound.
    timed_out:
        Whether the attempt was killed at its bound.
    classification:
        Wrapper classification value.
    gate_kind:
        Gate tier the attempt served.
    item_count:
        Number of models in the attempt's envelope.

    Returns
    -------
    dict[str, Any]
        Telemetry record in the wrapper's published shape.
    """

    return {
        "event": "codex-attempt",
        "attempt": attempt,
        "classification": classification,
        "detail": "",
        "returncode": -9 if timed_out else 0,
        "timed_out": timed_out,
        "started_at": "2026-07-30T00:00:00Z",
        "finished_at": "2026-07-30T00:10:00Z",
        "duration_seconds": duration,
        "attempt_budget_seconds": budget,
        "attempt_timeout_seconds": budget,
        "gate_kind": gate_kind,
        "item_count": item_count,
    }


def _write_telemetry(root: Path, records: list[dict[str, Any]]) -> Path:
    """Write one telemetry file beneath a nested request directory.

    Parameters
    ----------
    root:
        Work root the report scans.
    records:
        Records serialized one per line.

    Returns
    -------
    pathlib.Path
        The telemetry file written.
    """

    directory = root / "m_one" / "checker-fidelity"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "operator-telemetry.jsonl"
    lines = [json.dumps(record, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_completed_and_censored_samples_are_never_pooled(tmp_path: Path) -> None:
    """Timed-out durations are reported as lower bounds, never as completions.

    Parameters
    ----------
    tmp_path:
        Isolated work root.
    """

    _write_telemetry(
        tmp_path,
        [
            _attempt(duration=120.0),
            _attempt(attempt=2, duration=240.0),
            _attempt(
                attempt=3,
                duration=600.0,
                timed_out=True,
                classification="retryable-infrastructure",
            ),
            {"event": "operator-finished", "attempts": 3, "wall_seconds": 963.0},
        ],
    )

    report = build_report(iter_attempt_records([tmp_path]))

    assert report["attempts"] == 3
    assert report["completed"]["count"] == 2
    assert report["completed"]["max_seconds"] == 240.0
    assert report["censored_lower_bounds"]["count"] == 1
    assert report["censored_reaching_wall"] == 1
    assert report["censoring_rate"] == 1 / 3
    # Pooling would put the mean at 320s; the completed sample alone is 180s and is
    # the only figure a cap may honestly be derived from.
    assert report["completed"]["mean_seconds"] == 180.0
    assert report["classifications"] == {"retryable-infrastructure": 1, "success": 2}
    assert report["attempt_bounds_seconds"] == {"600": 3}


def test_workloads_of_different_size_are_reported_separately(tmp_path: Path) -> None:
    """A twenty-model metadata batch is never averaged into per-model fidelity calls.

    One flat attempt cap spans both, so the pooled median hides that the batch is
    the workload actually pressing on the cap.

    Parameters
    ----------
    tmp_path:
        Isolated work root.
    """

    _write_telemetry(
        tmp_path,
        [
            _attempt(duration=60.0),
            _attempt(attempt=2, duration=70.0),
            _attempt(
                attempt=3,
                duration=600.0,
                timed_out=True,
                classification="retryable-infrastructure",
                gate_kind="metadata_batch",
                item_count=20,
            ),
        ],
    )

    report = build_report(iter_attempt_records([tmp_path]))

    assert set(report["by_workload"]) == {"fidelity[1]", "metadata_batch[20]"}
    assert report["by_workload"]["fidelity[1]"]["censoring_rate"] == 0.0
    assert report["by_workload"]["fidelity[1]"]["completed"]["max_seconds"] == 70.0
    assert report["by_workload"]["metadata_batch[20]"]["censoring_rate"] == 1.0
    assert report["by_workload"]["metadata_batch[20]"]["completed"]["count"] == 0


def test_records_without_durations_are_counted_not_zeroed(tmp_path: Path) -> None:
    """A record predating duration capture is untimed, never a zero-second sample.

    Parameters
    ----------
    tmp_path:
        Isolated work root.
    """

    legacy = _attempt(duration=0.0)
    del legacy["duration_seconds"]
    del legacy["attempt_budget_seconds"]
    del legacy["attempt_timeout_seconds"]
    _write_telemetry(tmp_path, [legacy, _attempt(attempt=2, duration=90.0)])

    report = build_report(iter_attempt_records([tmp_path]))

    assert report["untimed_attempts"] == 1
    assert report["timed_attempts"] == 1
    assert report["completed"]["count"] == 1
    assert report["completed"]["min_seconds"] == 90.0


def test_empty_sample_reports_no_statistics_rather_than_zeros() -> None:
    """An absent sample yields ``None`` statistics, so nothing can be misread as fast."""

    empty = summarize([])

    assert empty["count"] == 0
    assert empty["median_seconds"] is None
    assert empty["max_seconds"] is None


def test_heavy_censoring_is_called_out_in_the_text_report(tmp_path: Path) -> None:
    """A censored-dominated sample warns instead of quoting a confident quantile.

    Parameters
    ----------
    tmp_path:
        Isolated work root.
    """

    records = [
        _attempt(
            attempt=index,
            duration=600.0,
            timed_out=True,
            classification="retryable-infrastructure",
        )
        for index in range(1, 10)
    ]
    records.append(_attempt(attempt=10, duration=95.0))
    _write_telemetry(tmp_path, records)

    text = format_report(build_report(iter_attempt_records([tmp_path])))

    assert "WARNING: heavy censoring" in text
    assert "reached_wall=9" in text


def test_main_refuses_a_missing_root(tmp_path: Path) -> None:
    """A nonexistent root is a refused report, not an empty one.

    Parameters
    ----------
    tmp_path:
        Isolated work root.
    """

    assert main(["--root", str(tmp_path / "absent")]) == 1
    _write_telemetry(tmp_path, [_attempt(duration=12.0)])
    assert main(["--root", str(tmp_path), "--json"]) == 0
