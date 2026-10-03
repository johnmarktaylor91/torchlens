"""Living regression gate for TorchLens benchmark JSON payloads.

Tolerance policy (R28-2). The per-row tolerance is::

    max(rel_tolerance * baseline_median_ms,
        iqr_multiplier * baseline_iqr_ms,
        floor_ms)

Only the BASELINE spread widens the tolerance: a noisy current run must never
widen the bar it is judged against (the pre-R28 ``max(baseline_iqr,
current_iqr)`` term let a contaminated run pass its own regressions). Two
candidate targets are drafted for the maintainer fork on the default ``rel_tolerance``:

- **Strict 2%** (``rel_tolerance=0.02``): catches real per-op regressions on
  quiet, thread-pinned hosts (R28 F5 measured <=5.2% worst-row same-commit
  drift, <2% typical, under the quiet protocol). Requires the R28-3 protocol
  (pinned threads, recorded env, fresh process per cell) to avoid false reds.
- **Lenient 10%** (``rel_tolerance=0.10``, current default): tolerant of
  contaminated runners but silently passes up to ~9.9% real regression.

The default stays 10% until the fork is decided; both are reachable via
``--rel-tolerance``.

Metric policy (b6-sol R28, T14-6). Rows are judged on PROCESS-CPU time
(``cpu_median_ms`` / ``cpu_iqr_ms``) whenever both sides carry it: wall clock
on a loaded box charges run-queue pressure to the code under test, which the
process-time samples do not. Rows where either side predates the CPU metrics
fall back to wall clock, disclosed per row via ``"metric"`` -- and that
fallback is NOT AUTHORITATIVE for TorchLens-owned rows (b6-sol R28 round 4:
a stale 196-row baseline with zero ``cpu_median_ms`` judged the ENTIRE run
on wall clock and the gate warned but PASSED, so the process-CPU ceiling was
unenforceable). By default a TorchLens-owned row judged on wall clock FAILS
the gate with instructions to regenerate the baseline; ``--allow-wall-clock-
only`` (``require_cpu_metrics=False``) is the explicit, disclosed opt-out
for legacy-baseline comparisons.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from benchmarks.op_ownership import is_torchlens_operation

SCHEMA = "torchlens.perf_gate.v1"
DEFAULT_REL_TOLERANCE = 0.10
DEFAULT_IQR_MULTIPLIER = 2.0
DEFAULT_FLOOR_MS = 0.5


def load_gate_json(path: Path) -> dict[str, Any]:
    """Load a benchmark gate JSON file.

    Parameters
    ----------
    path:
        JSON payload path.

    Returns
    -------
    dict[str, Any]
        Parsed benchmark payload.
    """

    payload = json.loads(path.read_text())
    validate_gate_payload(payload)
    return payload


def validate_gate_payload(payload: dict[str, Any]) -> None:
    """Validate the minimal P6 gate schema.

    Parameters
    ----------
    payload:
        Parsed benchmark payload.

    Raises
    ------
    ValueError
        If the payload is not a supported gate payload.
    """

    if payload.get("schema") not in {SCHEMA, None}:
        raise ValueError(f"Unsupported perf gate schema: {payload.get('schema')!r}")
    if "rows" not in payload or not isinstance(payload["rows"], list):
        raise ValueError("Perf gate payload must contain a rows list")
    if "environment" not in payload or not isinstance(payload["environment"], dict):
        raise ValueError("Perf gate payload must contain environment metadata")
    for row in payload["rows"]:
        _validate_row(row)


def normalize_gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with the P6 schema marker added.

    Parameters
    ----------
    payload:
        Benchmark payload produced by ``benchmarks.perf_suite``.

    Returns
    -------
    dict[str, Any]
        Normalized payload.
    """

    normalized = dict(payload)
    normalized.setdefault("schema", SCHEMA)
    return normalized


def compare_gate_payloads(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    floor_ms: float = DEFAULT_FLOOR_MS,
    require_cpu_metrics: bool = True,
) -> dict[str, Any]:
    """Compare current benchmark rows against a committed baseline.

    Parameters
    ----------
    baseline:
        Baseline gate payload.
    current:
        Current gate payload.
    rel_tolerance:
        Relative slowdown tolerance as a fraction of the baseline median.
    iqr_multiplier:
        Multiplier applied to the baseline IQR term of the tolerance.
    floor_ms:
        Absolute tolerance floor in milliseconds.
    require_cpu_metrics:
        When True (default), a TorchLens-owned row judged on the wall-clock
        fallback is gate-BLOCKING: the process-CPU policy cannot be enforced
        on the noisier metric it was added to replace (b6-sol R28 round 4).
        Pass False only for a disclosed legacy-baseline comparison.

    Returns
    -------
    dict[str, Any]
        Comparison summary with per-row verdicts. Gate-blocking lists:
        ``regressions``, ``status_failures``, ``unmatched_current_rows``
        (current rows with no baseline entry), ``missing_current_rows``
        (TorchLens-owned baseline rows that disappeared from the current
        run), and ``uncomparable_rows`` (matched ok TorchLens rows without
        usable timing metrics). ``unmatched_baseline_rows`` discloses
        vanished non-TorchLens rows without blocking.
    """

    validate_gate_payload(baseline)
    validate_gate_payload(current)
    baseline_by_key = {_row_key(row): row for row in baseline["rows"]}
    current_keys = {_row_key(row) for row in current["rows"]}
    checks: list[dict[str, Any]] = []
    unmatched_current: list[dict[str, str]] = []
    uncomparable: list[dict[str, str]] = []
    status_failures: list[dict[str, str]] = []
    regressions: list[dict[str, Any]] = []
    missing_current: list[dict[str, str]] = []
    unmatched_baseline: list[dict[str, str]] = []
    for key in baseline_by_key:
        if key in current_keys:
            continue
        if _is_torchlens_operation(key[2]):
            missing_current.append(_key_dict(key))
        else:
            unmatched_baseline.append(_key_dict(key))
    for row in current["rows"]:
        key = _row_key(row)
        base_row = baseline_by_key.get(key)
        if base_row is None:
            unmatched_current.append(_key_dict(key))
            continue
        current_status = str(row.get("status", "ok"))
        if _is_torchlens_operation(key[2]) and current_status != "ok":
            status_failures.append(_key_dict(key) | {"status": current_status})
            continue
        check = _compare_row(
            base_row,
            row,
            rel_tolerance=rel_tolerance,
            iqr_multiplier=iqr_multiplier,
            floor_ms=floor_ms,
        )
        if check is None:
            if (
                _is_torchlens_operation(key[2])
                and current_status == "ok"
                and str(base_row.get("status", "ok")) == "ok"
            ):
                uncomparable.append(_key_dict(key))
            continue
        checks.append(check)
        if not check["passed"]:
            regressions.append(check)
    # R28: the per-row wall-clock fallback exists for pre-CPU-metric
    # payloads, but a STALE baseline with zero cpu_* rows silently judged
    # the ENTIRE run on wall clock -- the noisier metric the CPU statistics
    # were added to replace -- and the gate WARNED but PASSED, so the
    # process-CPU policy was unenforceable (b6-sol R28 round 4). The
    # fallback is now non-authoritative for TorchLens-owned rows: by
    # default they join the blocking lists; the disclosed opt-out is
    # ``require_cpu_metrics=False`` / ``--allow-wall-clock-only``.
    wall_clock_rows = [check for check in checks if check.get("metric") == "wall_clock"]
    metric_fallback_blocking = [
        check
        for check in wall_clock_rows
        if require_cpu_metrics and _is_torchlens_operation(check["operation"])
    ]
    metric_degraded = bool(checks) and bool(wall_clock_rows)
    if metric_degraded:
        warnings.warn(
            f"perf gate judged {len(wall_clock_rows)}/{len(checks)} rows on WALL CLOCK "
            "because the baseline lacks process-CPU statistics (cpu_median_ms/"
            "cpu_iqr_ms). Regenerate the baseline with a current perf_suite run; "
            "wall-clock verdicts are load-sensitive"
            + (
                " and BLOCK this gate (pass --allow-wall-clock-only for a "
                "disclosed legacy comparison)."
                if metric_fallback_blocking
                else "."
            ),
            UserWarning,
            stacklevel=2,
        )
    passed = (
        not unmatched_current
        and not missing_current
        and not uncomparable
        and not status_failures
        and not regressions
        and not metric_fallback_blocking
    )
    return {
        "schema": SCHEMA,
        "passed": passed,
        "metric_degraded_to_wall_clock": metric_degraded,
        "wall_clock_row_count": len(wall_clock_rows),
        "wall_clock_fallback_blocking_rows": [
            _key_dict((check["model"], check["device"], check["operation"]))
            for check in metric_fallback_blocking
        ],
        "baseline_sha": baseline.get("source_sha")
        or baseline.get("environment", {}).get("torchlens_git_sha"),
        "current_sha": current.get("source_sha")
        or current.get("environment", {}).get("torchlens_git_sha"),
        "checks": checks,
        "unmatched_current_rows": unmatched_current,
        "missing_current_rows": missing_current,
        "unmatched_baseline_rows": unmatched_baseline,
        "uncomparable_rows": uncomparable,
        "status_failures": status_failures,
        "regressions": regressions,
        "tolerance_policy": (
            f"current - baseline <= max({rel_tolerance} * baseline_median_ms, "
            f"{iqr_multiplier} * baseline_iqr_ms, {floor_ms})"
        ),
    }


def write_comparison(path: Path, comparison: dict[str, Any]) -> None:
    """Write a comparison summary JSON file.

    Parameters
    ----------
    path:
        Destination path.
    comparison:
        Comparison payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")


def _validate_row(row: Any) -> None:
    """Validate one benchmark row.

    Parameters
    ----------
    row:
        Row object.

    Raises
    ------
    ValueError
        If the row is malformed.
    """

    if not isinstance(row, dict):
        raise ValueError("Perf gate rows must be objects")
    for key in ("model", "device", "operation", "status"):
        if key not in row:
            raise ValueError(f"Perf gate row missing {key!r}: {row!r}")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return the stable comparison key for a row.

    Parameters
    ----------
    row:
        Benchmark row.

    Returns
    -------
    tuple[str, str, str]
        ``(model, device, operation)`` key.
    """

    return (str(row["model"]), str(row["device"]), str(row["operation"]))


def _key_dict(key: tuple[str, str, str]) -> dict[str, str]:
    """Convert a row key to a JSON object.

    Parameters
    ----------
    key:
        Row key.

    Returns
    -------
    dict[str, str]
        JSON-serializable key.
    """

    return {"model": key[0], "device": key[1], "operation": key[2]}


def _timing(row: dict[str, Any], key: str) -> float | None:
    """Fetch a timing metric from a merged benchmark row.

    Parameters
    ----------
    row:
        Benchmark row.
    key:
        Timing metric name.

    Returns
    -------
    float | None
        Metric value.
    """

    value = row.get("passes", {}).get("timing", {}).get("timing", {}).get(key)
    return float(value) if isinstance(value, int | float) else None


def _compare_row(
    base_row: dict[str, Any],
    current_row: dict[str, Any],
    *,
    rel_tolerance: float,
    iqr_multiplier: float,
    floor_ms: float,
) -> dict[str, Any] | None:
    """Compare one matched row.

    Parameters
    ----------
    base_row:
        Baseline row.
    current_row:
        Current row.
    rel_tolerance:
        Relative slowdown tolerance as a fraction of the baseline median.
    iqr_multiplier:
        Multiplier applied to the baseline IQR term of the tolerance.
    floor_ms:
        Absolute tolerance floor in milliseconds.

    Returns
    -------
    dict[str, Any] | None
        Row comparison, or ``None`` when timing metrics are unavailable.
    """

    # Prefer process-CPU statistics whenever BOTH sides record them; wall
    # clock is only the legacy fallback for pre-CPU-metric payloads.
    metric = "process_cpu"
    baseline_median = _timing(base_row, "cpu_median_ms")
    current_median = _timing(current_row, "cpu_median_ms")
    baseline_iqr = _timing(base_row, "cpu_iqr_ms")
    current_iqr = _timing(current_row, "cpu_iqr_ms")
    if (
        baseline_median is None
        or current_median is None
        or baseline_iqr is None
        or current_iqr is None
    ):
        metric = "wall_clock"
        baseline_median = _timing(base_row, "median_ms")
        current_median = _timing(current_row, "median_ms")
        baseline_iqr = _timing(base_row, "iqr_ms")
        current_iqr = _timing(current_row, "iqr_ms")
    if (
        baseline_median is None
        or current_median is None
        or baseline_iqr is None
        or current_iqr is None
    ):
        return None
    tolerance = max(rel_tolerance * baseline_median, iqr_multiplier * baseline_iqr, floor_ms)
    delta = current_median - baseline_median
    key = _row_key(current_row)
    return {
        **_key_dict(key),
        "metric": metric,
        "baseline_median_ms": baseline_median,
        "current_median_ms": current_median,
        "baseline_iqr_ms": baseline_iqr,
        "current_iqr_ms": current_iqr,
        "delta_ms": delta,
        "ratio": current_median / baseline_median if baseline_median else None,
        "tolerance_ms": tolerance,
        "passed": delta <= tolerance,
    }


# Ownership classification lives in ONE module (b2 R41 round 5: this file
# and perf_suite.py carried divergence-prone copies while the classifier
# gates every blocking axis).
_is_torchlens_operation = is_torchlens_operation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=DEFAULT_REL_TOLERANCE,
        help="Relative slowdown tolerance as a fraction of the baseline median",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=DEFAULT_IQR_MULTIPLIER,
        help="Multiplier on the baseline IQR term of the tolerance",
    )
    parser.add_argument(
        "--floor-ms",
        type=float,
        default=DEFAULT_FLOOR_MS,
        help="Absolute tolerance floor in milliseconds",
    )
    parser.add_argument(
        "--allow-wall-clock-only",
        action="store_true",
        help=(
            "Permit TorchLens-owned rows judged on the wall-clock fallback to "
            "pass (disclosed legacy-baseline comparison); by default they BLOCK "
            "because the process-CPU policy cannot be enforced on wall clock"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the regression gate from the command line."""

    args = parse_args()
    comparison = compare_gate_payloads(
        load_gate_json(args.baseline),
        load_gate_json(args.current),
        rel_tolerance=args.rel_tolerance,
        iqr_multiplier=args.iqr_multiplier,
        floor_ms=args.floor_ms,
        require_cpu_metrics=not args.allow_wall_clock_only,
    )
    if args.out is not None:
        write_comparison(args.out, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
