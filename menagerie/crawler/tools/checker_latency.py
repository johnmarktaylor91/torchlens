"""Report the operator attempt-latency distribution from wrapper telemetry.

This exists so ``OPERATOR_ATTEMPT_TIMEOUT_SECONDS`` can be set from EVIDENCE rather
than from a synthetic probe. Every ``codex-attempt`` telemetry record carries its wall
duration and the bound it was held to, so the sample splits cleanly into two kinds:

* COMPLETED -- the attempt returned on its own. Its duration is an exact observation.
* CENSORED -- the attempt hit its wall and was killed. Its duration is a LOWER BOUND
  on the run it would have had, nothing more.

Those two kinds are reported separately and are never pooled into one mean, because
pooling them silently biases the estimate downward by exactly the amount the cap was
too small. A high censoring rate is the signal that the cap itself is unusable as
evidence: raise it and re-measure before quoting any quantile.

Usage::

    python -m menagerie.crawler.tools.checker_latency --root /path/to/work-root
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from menagerie.crawler.models import JsonObject

#: Telemetry file the checker wrapper publishes beside every request.
TELEMETRY_FILENAME = "operator-telemetry.jsonl"

#: The single telemetry event carrying one external attempt's timing.
ATTEMPT_EVENT = "codex-attempt"

#: An attempt within this many seconds of its bound is treated as having reached the
#: wall even if the recorded duration is fractionally short of it, which the kill and
#: drain path can produce.
WALL_TOLERANCE_SECONDS = 1.0


def iter_attempt_records(roots: Sequence[Path]) -> Iterator[JsonObject]:
    """Yield every attempt telemetry record found beneath the given roots.

    Parameters
    ----------
    roots:
        Directories searched recursively for ``operator-telemetry.jsonl``.

    Yields
    ------
    dict[str, Any]
        One ``codex-attempt`` record. Malformed lines are skipped: telemetry is a
        bounded ring buffer whose oldest line can legitimately be truncated.
    """

    for root in roots:
        if root.is_file():
            candidates: Iterable[Path] = [root]
        else:
            candidates = sorted(root.rglob(TELEMETRY_FILENAME))
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get("event") == ATTEMPT_EVENT:
                    yield value


def _duration(record: Mapping[str, Any]) -> Optional[float]:
    """Return one finite non-negative attempt duration, or ``None`` when absent.

    Parameters
    ----------
    record:
        Candidate attempt telemetry record.

    Returns
    -------
    float or None
        Recorded wall seconds. ``None`` for records written before durations were
        captured, which are reported as a separate untimed count rather than as zeros.
    """

    value = record.get("duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _bound(record: Mapping[str, Any]) -> Optional[float]:
    """Return the wall bound one attempt was held to, when it was recorded.

    Parameters
    ----------
    record:
        Candidate attempt telemetry record.

    Returns
    -------
    float or None
        Effective per-attempt budget in seconds.
    """

    value = record.get("attempt_budget_seconds", record.get("attempt_timeout_seconds"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def quantile(samples: Sequence[float], fraction: float) -> Optional[float]:
    """Return one nearest-rank quantile without an external numerical dependency.

    Parameters
    ----------
    samples:
        Ascending duration samples.
    fraction:
        Quantile in ``(0.0, 1.0]``.

    Returns
    -------
    float or None
        Nearest-rank value, or ``None`` for an empty sample.
    """

    if not samples:
        return None
    index = max(0, math.ceil(fraction * len(samples)) - 1)
    return samples[min(index, len(samples) - 1)]


def summarize(values: Iterable[float]) -> JsonObject:
    """Summarize one duration sample.

    Parameters
    ----------
    values:
        Non-negative duration samples in seconds.

    Returns
    -------
    dict[str, Any]
        Count and order statistics, with ``None`` statistics for an empty sample.
    """

    samples = sorted(float(value) for value in values)
    if not samples:
        return {
            "count": 0,
            "min_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "max_seconds": None,
            "mean_seconds": None,
        }
    return {
        "count": len(samples),
        "min_seconds": samples[0],
        "median_seconds": quantile(samples, 0.5),
        "p95_seconds": quantile(samples, 0.95),
        "p99_seconds": quantile(samples, 0.99),
        "max_seconds": samples[-1],
        "mean_seconds": math.fsum(samples) / len(samples),
    }


def _split(records: Iterable[Mapping[str, Any]]) -> JsonObject:
    """Split one record group into completed, censored, and untimed samples.

    Parameters
    ----------
    records:
        Attempt telemetry records belonging to one group.

    Returns
    -------
    dict[str, Any]
        Group report body.
    """

    completed: list[float] = []
    censored: list[float] = []
    reached_wall = 0
    untimed = 0
    classifications: Counter[str] = Counter()
    bounds: Counter[float] = Counter()
    total = 0
    for record in records:
        total += 1
        classifications[str(record.get("classification", "unknown"))] += 1
        bound = _bound(record)
        if bound is not None:
            bounds[bound] += 1
        duration = _duration(record)
        if duration is None:
            untimed += 1
            continue
        if bool(record.get("timed_out")):
            censored.append(duration)
            if bound is not None and duration >= bound - WALL_TOLERANCE_SECONDS:
                reached_wall += 1
        else:
            completed.append(duration)
    timed = len(completed) + len(censored)
    return {
        "attempts": total,
        "timed_attempts": timed,
        # Records predating duration capture. A nonzero count means the report is
        # built on a partial sample and must say so rather than quietly shrinking.
        "untimed_attempts": untimed,
        "censoring_rate": (len(censored) / timed) if timed else None,
        "completed": summarize(completed),
        # Lower bounds only. Reported so the sample's censoring is visible, never so
        # it can be pooled with the completed sample.
        "censored_lower_bounds": summarize(censored),
        "censored_reaching_wall": reached_wall,
        "attempt_bounds_seconds": {f"{bound:g}": count for bound, count in sorted(bounds.items())},
        "classifications": dict(sorted(classifications.items())),
    }


def _group_key(record: Mapping[str, Any]) -> str:
    """Return the workload group one attempt belongs to.

    Parameters
    ----------
    record:
        Attempt telemetry record.

    Returns
    -------
    str
        Gate kind with its batch size, so a one-model fidelity call is never
        averaged together with a twenty-model metadata batch.
    """

    gate_kind = record.get("gate_kind")
    kind = str(gate_kind) if isinstance(gate_kind, str) and gate_kind else "unknown"
    count = record.get("item_count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return f"{kind}[{count}]"
    return kind


def build_report(records: Iterable[Mapping[str, Any]]) -> JsonObject:
    """Build the completed-versus-censored attempt-latency report.

    Parameters
    ----------
    records:
        Attempt telemetry records.

    Returns
    -------
    dict[str, Any]
        Report object. ``completed`` is the only sample from which a latency
        quantile may honestly be quoted; ``censored`` durations are lower bounds.
        ``by_workload`` splits the same sample by gate kind and batch size, since
        one flat cap covers calls whose work differs by more than an order of
        magnitude.
    """

    collected = list(records)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in collected:
        groups.setdefault(_group_key(record), []).append(record)
    report = dict(_split(collected))
    report["by_workload"] = {key: _split(groups[key]) for key in sorted(groups)}
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Render one report as short operator-readable text.

    Parameters
    ----------
    report:
        Report object from :func:`build_report`.

    Returns
    -------
    str
        Human-readable summary.
    """

    completed = report["completed"]
    censored = report["censored_lower_bounds"]
    rate = report["censoring_rate"]
    lines = [
        f"attempts={report['attempts']} timed={report['timed_attempts']} "
        f"untimed={report['untimed_attempts']}",
        f"censoring_rate={'n/a' if rate is None else f'{rate:.1%}'} "
        f"(censored={censored['count']}, reached_wall={report['censored_reaching_wall']})",
    ]
    if completed["count"]:
        lines.append(
            f"completed n={completed['count']} "
            f"median={completed['median_seconds']:.1f}s "
            f"p95={completed['p95_seconds']:.1f}s "
            f"max={completed['max_seconds']:.1f}s"
        )
    else:
        lines.append("completed n=0 -- no uncensored observation; the cap is unmeasurable")
    if censored["count"]:
        lines.append(
            f"censored lower bounds n={censored['count']} max={censored['max_seconds']:.1f}s"
        )
    if rate is not None and rate > 0.2:
        lines.append(
            "WARNING: heavy censoring. Quantiles of the completed sample understate the "
            "true distribution; raise the attempt cap and re-measure."
        )
    workloads = report.get("by_workload") or {}
    for key in sorted(workloads):
        group = workloads[key]
        group_completed = group["completed"]
        median = group_completed["median_seconds"]
        maximum = group_completed["max_seconds"]
        lines.append(
            f"  {key}: n={group['attempts']} completed={group_completed['count']} "
            f"median={'n/a' if median is None else f'{median:.1f}s'} "
            f"max={'n/a' if maximum is None else f'{maximum:.1f}s'} "
            f"censored={group['censored_lower_bounds']['count']}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the attempt-latency report parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting one or more telemetry roots.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        type=Path,
        help=(
            "directory searched recursively for operator-telemetry.jsonl, or one such "
            "file; repeatable"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the full report object")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the read-only attempt-latency report command.

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
    roots = [Path(root).expanduser().resolve() for root in args.root]
    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        print(f"checker latency report failed: no such root {missing[0]}", file=sys.stderr)
        return 1
    report = build_report(iter_attempt_records(roots))
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
