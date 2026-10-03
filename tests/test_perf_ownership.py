"""Ownership census for the perf-gate operation classifier.

Post-fixwave-4 the ``is_torchlens_operation`` prefix classifier decides
every blocking axis of the perf gate (missing-row, status-failure,
uncomparable, and wall-clock-fallback blocking), and it lived as two
independent hand-copies in ``perf_gate.py`` and ``perf_suite.py`` with no
lockstep and no census (b2 R41 round 5, wave-born): a new op family named
outside the prefixes was silently FOREIGN on every axis. The copies now
alias one module (``benchmarks/op_ownership.py``); these tests pin the
single-authority shape and prove every operation the runner can emit
classifies as owned or declared-foreign, never unknown.
"""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks import op_ownership, perf_gate, perf_suite
from benchmarks.op_ownership import classify_operation, is_torchlens_operation

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def _runner_operations() -> set[str]:
    """Every operation literal the runner dispatches on."""

    source = (_BENCHMARKS_DIR / "perf_runner.py").read_text(encoding="utf-8")
    return set(re.findall(r'operation == "([a-z0-9_]+)"', source))


def test_every_runner_operation_is_classified() -> None:
    """No emitted operation may escape both prefix families."""

    operations = _runner_operations()
    assert len(operations) >= 20, f"runner operation census shrank: {sorted(operations)}"
    unknown = sorted(op for op in operations if classify_operation(op) == "unknown")
    assert not unknown, (
        f"operations no ownership family claims: {unknown} — an unclassified "
        "row escapes every gate-blocking axis (missing/failed/uncomparable/"
        "wall-fallback); add its prefix to TORCHLENS_OPERATION_PREFIXES or "
        "FOREIGN_OPERATION_PREFIXES in benchmarks/op_ownership.py"
    )


def test_gate_and_suite_share_one_classifier() -> None:
    """Both consumers alias the op_ownership authority, not private copies."""

    assert perf_gate._is_torchlens_operation is is_torchlens_operation
    assert perf_suite._is_torchlens_operation is is_torchlens_operation


def test_classifier_is_red_capable() -> None:
    """Known owned/foreign/unknown shapes classify as expected."""

    assert classify_operation("tl_trace") == "owned"
    assert is_torchlens_operation("tl_trace")
    assert classify_operation("peer_nnsight") == "foreign"
    assert not is_torchlens_operation("peer_nnsight")
    assert classify_operation("record_span_probe") == "unknown"
    assert not is_torchlens_operation("record_span_probe")
    overlap = [
        op
        for op in op_ownership.TORCHLENS_OPERATION_PREFIXES
        if op.startswith(op_ownership.FOREIGN_OPERATION_PREFIXES)
    ]
    assert not overlap, f"prefix families overlap: {overlap}"


def _rerun_row(operation: str, timing: dict[str, float]) -> dict:
    return {
        "model": "tinynet",
        "device": "cpu",
        "operation": operation,
        "passes": {"timing": {"timing": dict(timing)}},
    }


def test_rerun_tolerance_is_cpu_authoritative_and_reference_widened() -> None:
    """The rerun stability check uses process-CPU and run1's IQR only.

    b6 R28 round 5: the check read wall medians and took
    ``2 * max(iqr_run1, iqr_run2)``, so a loaded box's wall drift dominated
    the verdict and the second run WIDENED its own acceptance band with its
    own noise — an unstable rerun could never fail the stability check.
    """

    from benchmarks.perf_suite import _check_rerun_tolerance

    # CPU fields present: verdict keys on cpu_median_ms even when the wall
    # numbers scream instability.
    quiet_cpu = {
        "median_ms": 100.0,
        "iqr_ms": 1.0,
        "cpu_median_ms": 100.0,
        "cpu_iqr_ms": 1.0,
    }
    noisy_wall_same_cpu = {
        "median_ms": 400.0,
        "iqr_ms": 50.0,
        "cpu_median_ms": 101.0,
        "cpu_iqr_ms": 1.0,
    }
    result = _check_rerun_tolerance(
        [_rerun_row("tl_trace", quiet_cpu)], [_rerun_row("tl_trace", noisy_wall_same_cpu)]
    )
    assert result["checks"][0]["metric"] == "cpu_median_ms"
    assert result["passed"], result

    # Reference-side IQR only: a wildly noisy SECOND run must not widen its
    # own band. run1 is tight (iqr 1ms), run2 drifts 60ms with a huge own-IQR
    # that under the old max() rule would have self-blessed the drift.
    drifted = {
        "median_ms": 160.0,
        "iqr_ms": 100.0,
        "cpu_median_ms": 160.0,
        "cpu_iqr_ms": 100.0,
    }
    result = _check_rerun_tolerance(
        [_rerun_row("tl_trace", quiet_cpu)], [_rerun_row("tl_trace", drifted)]
    )
    assert not result["passed"], result

    # Legacy rows without CPU fields fall back to wall, disclosed via metric.
    legacy = {"median_ms": 100.0, "iqr_ms": 1.0}
    result = _check_rerun_tolerance(
        [_rerun_row("tl_trace", legacy)], [_rerun_row("tl_trace", legacy)]
    )
    assert result["checks"][0]["metric"] == "median_ms"
    assert result["passed"]
