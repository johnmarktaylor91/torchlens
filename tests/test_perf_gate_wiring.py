"""Smoke-tier wiring test for ``benchmarks/perf_gate.py`` (b10 R78 round-3).

The perf gate existed with NO smoke-tier exercise: a refactor could break its
import, its baseline files, or its comparison math and nothing red would show
until someone ran a real benchmark. This file imports the gate machinery,
checks the committed baseline payloads still parse under the gate's own
schema validation, and proves the comparison logic on synthetic numbers —
including the red-capable direction: a synthetic regression MUST trip the
gate. No actual benchmarks run here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PERF_GATE_PATH = _REPO_ROOT / "benchmarks" / "perf_gate.py"
_BASELINE_DIR = _REPO_ROOT / "benchmarks" / "perf_baselines"


@pytest.fixture(scope="module")
def perf_gate() -> Any:
    """Import ``benchmarks/perf_gate.py`` as a standalone module."""

    spec = importlib.util.spec_from_file_location("_tl_perf_gate_wiring", _PERF_GATE_PATH)
    assert spec is not None and spec.loader is not None, _PERF_GATE_PATH
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _row(module: Any, operation: str, median_ms: float, iqr_ms: float) -> dict[str, Any]:
    """Build one synthetic benchmark row carrying both metric families."""

    del module
    timing = {
        "cpu_median_ms": median_ms,
        "cpu_iqr_ms": iqr_ms,
        "median_ms": median_ms,
        "iqr_ms": iqr_ms,
    }
    return {
        "model": "synthetic_mlp",
        "device": "cpu",
        "operation": operation,
        "status": "ok",
        "passes": {"timing": {"timing": timing}},
    }


def _payload(module: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap rows in a schema-valid gate payload."""

    return {"schema": module.SCHEMA, "environment": {"synthetic": True}, "rows": rows}


def test_perf_gate_committed_baselines_parse(perf_gate: Any) -> None:
    """Every committed baseline file passes the gate's own schema validation."""

    baseline_files = sorted(_BASELINE_DIR.glob("*.json"))
    assert baseline_files, f"no committed perf baselines under {_BASELINE_DIR}"
    for path in baseline_files:
        payload = perf_gate.load_gate_json(path)
        assert payload["rows"], f"baseline {path.name} carries no rows"


def test_perf_gate_synthetic_regression_trips(perf_gate: Any) -> None:
    """A synthetic 2x slowdown on a TorchLens-owned row FAILS the gate."""

    baseline = _payload(perf_gate, [_row(perf_gate, "trace_forward", 100.0, 1.0)])
    current = _payload(perf_gate, [_row(perf_gate, "trace_forward", 200.0, 1.0)])
    comparison = perf_gate.compare_gate_payloads(baseline, current)
    assert comparison["passed"] is False
    assert len(comparison["regressions"]) == 1
    check = comparison["regressions"][0]
    assert check["operation"] == "trace_forward"
    assert check["metric"] == "process_cpu"
    assert check["delta_ms"] == pytest.approx(100.0)


def test_perf_gate_within_tolerance_passes(perf_gate: Any) -> None:
    """A within-tolerance drift passes; the tolerance formula is honored.

    tolerance = max(rel * baseline_median, iqr_mult * baseline_iqr, floor)
              = max(0.10 * 100, 2.0 * 1, 0.5) = 10ms.
    """

    baseline = _payload(perf_gate, [_row(perf_gate, "trace_forward", 100.0, 1.0)])
    within = _payload(perf_gate, [_row(perf_gate, "trace_forward", 109.0, 1.0)])
    just_over = _payload(perf_gate, [_row(perf_gate, "trace_forward", 111.0, 1.0)])
    assert perf_gate.compare_gate_payloads(baseline, within)["passed"] is True
    assert perf_gate.compare_gate_payloads(baseline, just_over)["passed"] is False


def test_perf_gate_blocks_vanished_and_failed_torchlens_rows(perf_gate: Any) -> None:
    """Missing and non-ok TorchLens-owned rows block the gate, not just slowdowns."""

    baseline = _payload(
        perf_gate,
        [
            _row(perf_gate, "trace_forward", 100.0, 1.0),
            _row(perf_gate, "tl_summary", 10.0, 0.5),
        ],
    )
    vanished = _payload(perf_gate, [_row(perf_gate, "trace_forward", 100.0, 1.0)])
    comparison = perf_gate.compare_gate_payloads(baseline, vanished)
    assert comparison["passed"] is False
    assert comparison["missing_current_rows"] == [
        {"model": "synthetic_mlp", "device": "cpu", "operation": "tl_summary"}
    ]

    failed_row = _row(perf_gate, "tl_summary", 10.0, 0.5)
    failed_row["status"] = "error"
    failed = _payload(perf_gate, [_row(perf_gate, "trace_forward", 100.0, 1.0), failed_row])
    comparison = perf_gate.compare_gate_payloads(baseline, failed)
    assert comparison["passed"] is False
    assert comparison["status_failures"] == [
        {
            "model": "synthetic_mlp",
            "device": "cpu",
            "operation": "tl_summary",
            "status": "error",
        }
    ]


def test_perf_gate_wall_clock_fallback_is_not_authoritative(perf_gate: Any) -> None:
    """A TorchLens row judged on wall clock BLOCKS the gate by default.

    b6-sol R28 round 4: the committed 196-row baseline carried zero
    ``cpu_median_ms``, every row fell back to wall clock, and the gate
    warned but PASSED -- the process-CPU policy was unenforceable. The
    fallback verdict must not bless the run; the explicit
    ``require_cpu_metrics=False`` opt-out is the only sanctioned legacy
    comparison path, and it stays fully disclosed.
    """

    baseline_row = _row(perf_gate, "trace_forward", 100.0, 1.0)
    del baseline_row["passes"]["timing"]["timing"]["cpu_median_ms"]
    del baseline_row["passes"]["timing"]["timing"]["cpu_iqr_ms"]
    baseline = _payload(perf_gate, [baseline_row])
    current = _payload(perf_gate, [_row(perf_gate, "trace_forward", 101.0, 1.0)])
    with pytest.warns(UserWarning, match="WALL CLOCK"):
        comparison = perf_gate.compare_gate_payloads(baseline, current)
    assert comparison["metric_degraded_to_wall_clock"] is True
    assert comparison["checks"][0]["metric"] == "wall_clock"
    assert comparison["passed"] is False, "wall-only TL rows must not pass by default"
    assert comparison["wall_clock_fallback_blocking_rows"] == [
        {"model": "synthetic_mlp", "device": "cpu", "operation": "trace_forward"}
    ]
    with pytest.warns(UserWarning, match="WALL CLOCK"):
        legacy = perf_gate.compare_gate_payloads(baseline, current, require_cpu_metrics=False)
    assert legacy["passed"] is True, "the disclosed legacy opt-out still compares"
    assert legacy["metric_degraded_to_wall_clock"] is True
    assert legacy["wall_clock_fallback_blocking_rows"] == []


def test_perf_gate_wall_clock_fallback_on_foreign_rows_does_not_block(perf_gate: Any) -> None:
    """Non-TorchLens comparison rows (raw torch baselines) may stay wall-only."""

    baseline_row = _row(perf_gate, "torch_forward", 100.0, 1.0)
    del baseline_row["passes"]["timing"]["timing"]["cpu_median_ms"]
    del baseline_row["passes"]["timing"]["timing"]["cpu_iqr_ms"]
    tl_base = _row(perf_gate, "trace_forward", 100.0, 1.0)
    baseline = _payload(perf_gate, [baseline_row, tl_base])
    current = _payload(
        perf_gate,
        [
            _row(perf_gate, "torch_forward", 101.0, 1.0),
            _row(perf_gate, "trace_forward", 101.0, 1.0),
        ],
    )
    with pytest.warns(UserWarning, match="WALL CLOCK"):
        comparison = perf_gate.compare_gate_payloads(baseline, current)
    assert comparison["passed"] is True
    assert comparison["wall_clock_fallback_blocking_rows"] == []
