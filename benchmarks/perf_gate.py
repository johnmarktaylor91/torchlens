"""Living regression gate for TorchLens benchmark JSON payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "torchlens.perf_gate.v1"


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
) -> dict[str, Any]:
    """Compare current benchmark rows against a committed baseline.

    Parameters
    ----------
    baseline:
        Baseline gate payload.
    current:
        Current gate payload.

    Returns
    -------
    dict[str, Any]
        Comparison summary with per-row verdicts.
    """

    validate_gate_payload(baseline)
    validate_gate_payload(current)
    baseline_by_key = {_row_key(row): row for row in baseline["rows"]}
    checks: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    status_failures: list[dict[str, str]] = []
    regressions: list[dict[str, Any]] = []
    for row in current["rows"]:
        key = _row_key(row)
        base_row = baseline_by_key.get(key)
        if base_row is None:
            missing.append(_key_dict(key))
            continue
        current_status = str(row.get("status", "ok"))
        if _is_torchlens_operation(key[2]) and current_status != "ok":
            status_failures.append(_key_dict(key) | {"status": current_status})
            continue
        check = _compare_row(base_row, row)
        if check is None:
            continue
        checks.append(check)
        if not check["passed"]:
            regressions.append(check)
    passed = not missing and not status_failures and not regressions
    return {
        "schema": SCHEMA,
        "passed": passed,
        "baseline_sha": baseline.get("source_sha")
        or baseline.get("environment", {}).get("torchlens_git_sha"),
        "current_sha": current.get("source_sha")
        or current.get("environment", {}).get("torchlens_git_sha"),
        "checks": checks,
        "missing_baseline_rows": missing,
        "status_failures": status_failures,
        "regressions": regressions,
        "tolerance_policy": "current - baseline <= max(0.10 * baseline_median_ms, "
        "2 * max(baseline_iqr_ms, current_iqr_ms), 0.5)",
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


def _compare_row(base_row: dict[str, Any], current_row: dict[str, Any]) -> dict[str, Any] | None:
    """Compare one matched row.

    Parameters
    ----------
    base_row:
        Baseline row.
    current_row:
        Current row.

    Returns
    -------
    dict[str, Any] | None
        Row comparison, or ``None`` when timing metrics are unavailable.
    """

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
    tolerance = max(0.10 * baseline_median, 2 * max(baseline_iqr, current_iqr), 0.5)
    delta = current_median - baseline_median
    key = _row_key(current_row)
    return {
        **_key_dict(key),
        "baseline_median_ms": baseline_median,
        "current_median_ms": current_median,
        "baseline_iqr_ms": baseline_iqr,
        "current_iqr_ms": current_iqr,
        "delta_ms": delta,
        "ratio": current_median / baseline_median if baseline_median else None,
        "tolerance_ms": tolerance,
        "passed": delta <= tolerance,
    }


def _is_torchlens_operation(operation: str) -> bool:
    """Return whether an operation is owned by TorchLens.

    Parameters
    ----------
    operation:
        Benchmark operation identifier.

    Returns
    -------
    bool
        True when failures should be gate-blocking.
    """

    return operation.startswith(
        (
            "aux_",
            "fastlog_",
            "first_capture",
            "global_wrap",
            "raw_global",
            "raw_target",
            "raw_tl",
            "rerun_",
            "tl_",
            "trace_",
        )
    )


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
    return parser.parse_args()


def main() -> None:
    """Run the regression gate from the command line."""

    args = parse_args()
    comparison = compare_gate_payloads(
        load_gate_json(args.baseline),
        load_gate_json(args.current),
    )
    if args.out is not None:
        write_comparison(args.out, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
