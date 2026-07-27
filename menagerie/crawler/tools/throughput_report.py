"""Build read-only timing and late-campaign slope reports for crawler campaigns."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths

CampaignPaths = Mapping[str, tuple[Path, Path]]


def _parse_timestamp(value: object) -> Optional[datetime]:
    """Parse one UTC timestamp, returning ``None`` for absent timing data.

    Parameters
    ----------
    value:
        Candidate RFC 3339 timestamp.

    Returns
    -------
    datetime or None
        Parsed timezone-aware timestamp when valid.
    """

    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(started: object, finished: object) -> Optional[float]:
    """Return a non-negative timestamp delta when both endpoints are valid.

    Parameters
    ----------
    started, finished:
        Candidate RFC 3339 timestamps.

    Returns
    -------
    float or None
        Non-negative elapsed seconds, or ``None`` for invalid endpoints.
    """

    start = _parse_timestamp(started)
    finish = _parse_timestamp(finished)
    if start is None or finish is None:
        return None
    duration = (finish - start).total_seconds()
    return duration if math.isfinite(duration) and duration >= 0.0 else None


def _summary(values: Iterable[float]) -> JsonObject:
    """Summarize one timing sample without external numerical dependencies.

    Parameters
    ----------
    values:
        Non-negative elapsed-second samples.

    Returns
    -------
    dict[str, Any]
        Count, total, mean, and nearest-rank p95.
    """

    samples = sorted(float(value) for value in values)
    if not samples:
        return {"count": 0, "total_seconds": 0.0, "mean_seconds": None, "p95_seconds": None}
    p95_index = max(0, math.ceil(0.95 * len(samples)) - 1)
    total = math.fsum(samples)
    return {
        "count": len(samples),
        "total_seconds": total,
        "mean_seconds": total / len(samples),
        "p95_seconds": samples[p95_index],
    }


def fit_linear_slope(samples: Iterable[tuple[int, float]]) -> Optional[float]:
    """Fit elapsed seconds against terminal count with ordinary least squares.

    Parameters
    ----------
    samples:
        ``(terminal_count, per_model_seconds)`` observations.

    Returns
    -------
    float or None
        Seconds added per terminal, or ``None`` without two distinct x values.
    """

    values = [(float(x), float(y)) for x, y in samples]
    if len(values) < 2:
        return None
    mean_x = math.fsum(x for x, _y in values) / len(values)
    mean_y = math.fsum(y for _x, y in values) / len(values)
    denominator = math.fsum((x - mean_x) ** 2 for x, _y in values)
    if denominator == 0.0:
        return None
    numerator = math.fsum((x - mean_x) * (y - mean_y) for x, y in values)
    return numerator / denominator


def _attempt_timings(records: Iterable[Mapping[str, Any]]) -> Mapping[str, JsonObject]:
    """Aggregate attempt durations by reducer stage.

    Parameters
    ----------
    records:
        Canonical attempt rows.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Stage to duration summary.
    """

    samples: dict[str, list[float]] = defaultdict(list)
    for record in records:
        duration = _duration_seconds(record.get("started_at"), record.get("finished_at"))
        if duration is not None:
            samples[str(record.get("stage"))].append(duration)
    return {stage: _summary(values) for stage, values in sorted(samples.items())}


def _gate_timings(records: Iterable[Mapping[str, Any]]) -> Mapping[str, JsonObject]:
    """Aggregate checker durations by gate kind.

    Parameters
    ----------
    records:
        Canonical gate rows.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Gate kind to duration summary.
    """

    samples: dict[str, list[float]] = defaultdict(list)
    for record in records:
        checker = record.get("checker")
        if not isinstance(checker, Mapping):
            continue
        duration = _duration_seconds(checker.get("started_at"), checker.get("finished_at"))
        if duration is not None:
            samples[str(record.get("gate_kind"))].append(duration)
    return {kind: _summary(values) for kind, values in sorted(samples.items())}


def _operational_deltas(records: Iterable[Mapping[str, Any]]) -> Mapping[str, JsonObject]:
    """Aggregate elapsed time since the prior operational event by event kind.

    Parameters
    ----------
    records:
        Canonical operational rows in ledger order.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Event kind to inter-event delta summary.
    """

    samples: dict[str, list[float]] = defaultdict(list)
    prior: Optional[datetime] = None
    for record in records:
        current = _parse_timestamp(record.get("created_at"))
        if current is not None and prior is not None:
            duration = (current - prior).total_seconds()
            if math.isfinite(duration) and duration >= 0.0:
                samples[str(record.get("event_kind"))].append(duration)
        if current is not None:
            prior = current
    return {kind: _summary(values) for kind, values in sorted(samples.items())}


def _forward_wall_summary(records: Iterable[Mapping[str, Any]]) -> JsonObject:
    """Summarize parent-observed wall time for forward-stage attempts.

    Parameters
    ----------
    records:
        Canonical attempt rows.

    Returns
    -------
    dict[str, Any]
        Forward wall-time summary.
    """

    values: list[float] = []
    for record in records:
        if record.get("stage") != "forward":
            continue
        observation = record.get("supervisor_observation")
        wall_seconds = observation.get("wall_seconds") if isinstance(observation, Mapping) else None
        if (
            isinstance(wall_seconds, (int, float))
            and not isinstance(wall_seconds, bool)
            and math.isfinite(float(wall_seconds))
            and float(wall_seconds) >= 0.0
        ):
            values.append(float(wall_seconds))
    return _summary(values)


def _read_hot_path(path: Path) -> list[JsonObject]:
    """Read optional W1.4 hot-path instrumentation without schema assumptions.

    Parameters
    ----------
    path:
        Local instrumentation JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Structurally usable metric rows.
    """

    rows = scan_jsonl(path, validate=False)
    return [
        row
        for row in rows
        if row.get("metric") == "final-authority-per-model-seconds-vs-terminal-count"
        and isinstance(row.get("terminal_count"), int)
        and not isinstance(row.get("terminal_count"), bool)
        and isinstance(row.get("elapsed_seconds"), (int, float))
        and not isinstance(row.get("elapsed_seconds"), bool)
        and math.isfinite(float(row["elapsed_seconds"]))
        and float(row["elapsed_seconds"]) >= 0.0
    ]


def _campaign_report(records_root: Path, runtime_root: Path) -> JsonObject:
    """Build one campaign's timing and slope projection.

    Parameters
    ----------
    records_root, runtime_root:
        Canonical campaign records and private runtime roots.

    Returns
    -------
    dict[str, Any]
        Read-only timing projection.
    """

    ledgers = default_ledger_paths(records_root)
    attempts = scan_jsonl(ledgers.attempts)
    gates = scan_jsonl(ledgers.gates)
    operational = scan_jsonl(canonical_operational_ledger_path(ledgers.models))
    hot_path = _read_hot_path(runtime_root / "instrumentation" / "ledger-hot-path.jsonl")
    slope_samples = [
        (int(row["terminal_count"]), float(row["elapsed_seconds"])) for row in hot_path
    ]
    slope = fit_linear_slope(slope_samples)
    cache_enabled = sum(row.get("checkpoint_cache_enabled") is True for row in hot_path)
    cache_hits = sum(row.get("checkpoint_cache_hit") is True for row in hot_path)
    return {
        "attempt_stage_seconds": _attempt_timings(attempts),
        "checker_gate_seconds": _gate_timings(gates),
        "operational_inter_event_seconds": _operational_deltas(operational),
        "forward_wall_seconds": _forward_wall_summary(attempts),
        "hot_path": {
            "metric": "final-authority-per-model-seconds-vs-terminal-count",
            "sample_count": len(hot_path),
            "slope_seconds_per_terminal": slope,
            "positive_slope_detected": slope is not None and slope > 0.0,
            "checkpoint_cache_enabled_samples": cache_enabled,
            "checkpoint_cache_hit_samples": cache_hits,
        },
    }


def build_throughput_report(campaign_paths: CampaignPaths) -> JsonObject:
    """Build the four-campaign read-only throughput and R5 alert report.

    Parameters
    ----------
    campaign_paths:
        Campaign ID to ``(records_root, runtime_root)`` paths.

    Returns
    -------
    dict[str, Any]
        Per-campaign timing projections and aggregate positive-slope alert.
    """

    campaigns = {
        campaign_id: _campaign_report(records_root, runtime_root)
        for campaign_id, (records_root, runtime_root) in sorted(campaign_paths.items())
    }
    positive = sorted(
        campaign_id
        for campaign_id, report in campaigns.items()
        if report["hot_path"]["positive_slope_detected"] is True
    )
    return {
        "format": "menagerie.crawler.throughput-report.v1",
        "campaigns": campaigns,
        "risk_r5": {
            "alert": bool(positive),
            "positive_slope_campaigns": positive,
            "interpretation": (
                "positive slope means per-model final-authority time grows with terminal count"
            ),
        },
    }


def _campaign_argument(value: str) -> tuple[str, tuple[Path, Path]]:
    """Parse ``CAMPAIGN_ID=CLONE_ROOT`` into report paths.

    Parameters
    ----------
    value:
        Command-line campaign binding.

    Returns
    -------
    tuple[str, tuple[Path, Path]]
        Campaign ID and canonical/private roots.
    """

    campaign_id, separator, raw_root = value.partition("=")
    if not separator or not campaign_id or not raw_root:
        raise argparse.ArgumentTypeError("campaign must be CAMPAIGN_ID=CLONE_ROOT")
    clone_root = Path(raw_root).expanduser().resolve()
    return (
        campaign_id,
        (
            clone_root / "menagerie" / "crawler" / "records",
            clone_root / ".crawl-local",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the throughput-report command parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for campaign clone roots and optional output.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        type=_campaign_argument,
        help="repeat as CAMPAIGN_ID=CLONE_ROOT",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the read-only throughput report command.

    Parameters
    ----------
    argv:
        Optional arguments excluding the executable name.

    Returns
    -------
    int
        Zero on success and one on a refused report.
    """

    args = build_parser().parse_args(argv)
    campaign_paths = dict(args.campaign)
    if len(campaign_paths) != len(args.campaign):
        print("throughput report failed: duplicate campaign argument", file=sys.stderr)
        return 1
    try:
        report = build_throughput_report(campaign_paths)
        payload = canonical_json_bytes(report) + b"\n"
        if args.output is not None:
            atomic_replace_bytes(args.output, payload)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"throughput report failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
