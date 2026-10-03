"""Measurement-method tests for the benchmark runner and regression gate.

These are the R28 seed tests: the runner must speak the current ``save=``
predicate API (not the removed ``keep_op=``/``keep_module=`` spelling), the
gate must detect baseline rows that disappear from the current run, timing
must record CPU time alongside wall time, and the memory pass must report
phase-local peaks.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from benchmarks.perf_gate import compare_gate_payloads, normalize_gate_payload
from benchmarks.perf_runner import (
    _operation,
    _run_memory,
    _run_timing,
    _select_fastlog_names,
)


def _tiny_model() -> tuple[nn.Module, torch.Tensor]:
    """Build a tiny deterministic model/input pair for runner smoke checks.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Evaluation-mode model and matching input tensor.
    """

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    model.eval()
    return model, torch.randn(1, 4)


def _row(
    model: str,
    device: str,
    operation: str,
    median_ms: float,
    *,
    iqr_ms: float = 1.0,
    status: str = "ok",
) -> dict[str, object]:
    """Build one synthetic benchmark row.

    Parameters
    ----------
    model:
        Model identifier.
    device:
        Device identifier.
    operation:
        Operation identifier.
    median_ms:
        Median timing.
    iqr_ms:
        Interquartile range timing.
    status:
        Row status.

    Returns
    -------
    dict[str, object]
        Benchmark row.
    """

    # Modern payloads carry BOTH metric families; the gate judges on the
    # process-CPU pair and wall-only rows are non-authoritative (b6-sol R28
    # round 4), so the synthetic rows mirror what perf_runner emits.
    return {
        "model": model,
        "device": device,
        "operation": operation,
        "label": operation,
        "status": status,
        "passes": {
            "timing": {
                "timing": {
                    "median_ms": median_ms,
                    "iqr_ms": iqr_ms,
                    "cpu_median_ms": median_ms,
                    "cpu_iqr_ms": iqr_ms,
                }
            }
        },
    }


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build a synthetic gate payload.

    Parameters
    ----------
    rows:
        Gate rows.

    Returns
    -------
    dict[str, object]
        Gate payload.
    """

    return normalize_gate_payload(
        {
            "date": "2026-08-14",
            "environment": {"torchlens_git_sha": "abc1234"},
            "rows": rows,
        }
    )


def test_runner_fastlog_zero_operation_executes_on_current_api() -> None:
    """The fastlog_zero cell runs against the current ``save=`` predicate API."""

    model, x = _tiny_model()
    state: dict[str, object] = {}

    fn = _operation("fastlog_zero", model, x, "cpu", state)
    recording = fn()

    assert recording is not None
    assert "save=" in str(state["fastlog_zero_policy"])


@pytest.mark.smoke
def test_runner_fastlog_selectivity_operations_execute_on_current_api() -> None:
    """Selectivity cells (dry_run + record) run against the ``save=`` API."""

    model, x = _tiny_model()
    state: dict[str, object] = {}

    fn = _operation("fastlog_op_10", model, x, "cpu", state)
    recording = fn()

    assert recording is not None
    assert state["fastlog_selected_func_names"]


def test_runner_fastlog_halt_operation_executes_on_current_api() -> None:
    """Halt-fraction cells derive their halt index through the ``save=`` API."""

    model, x = _tiny_model()
    state: dict[str, object] = {}

    fn = _operation("fastlog_halt_50", model, x, "cpu", state)
    recording = fn()

    assert recording is not None
    assert state["fastlog_halt_raw_index"] >= 1


def test_select_fastlog_names_returns_nonempty_selection() -> None:
    """Name selection sees real op events from the predicate dry run."""

    model, x = _tiny_model()

    names = _select_fastlog_names(model, x, 0.5)

    assert names
    assert all(isinstance(name, str) and name for name in names)


def test_gate_fails_when_baseline_row_disappears_from_current() -> None:
    """A TorchLens baseline row absent from the current run blocks the gate."""

    baseline = _payload(
        [
            _row("resnet18", "cpu", "tl_trace", 100.0),
            _row("resnet18", "cpu", "fastlog_op_10", 40.0),
        ]
    )
    current = _payload([_row("resnet18", "cpu", "tl_trace", 100.0)])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["missing_current_rows"] == [
        {"model": "resnet18", "device": "cpu", "operation": "fastlog_op_10"}
    ]


def test_gate_reports_non_torchlens_disappearance_without_blocking() -> None:
    """A vanished peer row is disclosed but does not block the gate."""

    baseline = _payload(
        [
            _row("resnet18", "cpu", "tl_trace", 100.0),
            _row("resnet18", "cpu", "peer_baukit", 40.0),
        ]
    )
    current = _payload([_row("resnet18", "cpu", "tl_trace", 100.0)])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is True
    assert comparison["unmatched_baseline_rows"] == [
        {"model": "resnet18", "device": "cpu", "operation": "peer_baukit"}
    ]


def test_gate_current_only_rows_stay_blocking_under_new_field_name() -> None:
    """Current rows with no baseline entry still block, under an honest name."""

    baseline = _payload([_row("resnet18", "cpu", "tl_trace", 100.0)])
    current = _payload(
        [
            _row("resnet18", "cpu", "tl_trace", 100.0),
            _row("resnet18", "cpu", "fastlog_op_50", 40.0),
        ]
    )

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["unmatched_current_rows"] == [
        {"model": "resnet18", "device": "cpu", "operation": "fastlog_op_50"}
    ]
    assert "missing_baseline_rows" not in comparison


def test_gate_tolerance_ignores_current_run_iqr() -> None:
    """A noisy current run cannot widen its own regression tolerance."""

    baseline = _payload([_row("resnet18", "cpu", "tl_trace", 100.0, iqr_ms=1.0)])
    current = _payload([_row("resnet18", "cpu", "tl_trace", 115.0, iqr_ms=50.0)])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["checks"][0]["tolerance_ms"] == 10.0


def test_gate_tolerance_parameters_are_configurable() -> None:
    """The relative tolerance and floor are explicit knobs, not constants."""

    baseline = _payload([_row("resnet18", "cpu", "tl_trace", 100.0, iqr_ms=0.1)])
    current = _payload([_row("resnet18", "cpu", "tl_trace", 103.0, iqr_ms=0.1)])

    strict = compare_gate_payloads(baseline, current, rel_tolerance=0.02, floor_ms=0.1)
    lenient = compare_gate_payloads(baseline, current, rel_tolerance=0.10, floor_ms=0.5)

    assert strict["passed"] is False
    assert lenient["passed"] is True
    assert "0.02" in strict["tolerance_policy"]


def test_gate_fails_when_no_matched_rows_are_comparable() -> None:
    """Matched rows with no usable timing metrics cannot silently pass."""

    baseline = _payload([_row("resnet18", "cpu", "tl_trace", 100.0)])
    current = _payload([_row("resnet18", "cpu", "tl_trace", 100.0)])
    # Strip BOTH metric families: with only one gone the row is still
    # comparable through the other (cpu preferred, wall as legacy fallback).
    del current["rows"][0]["passes"]["timing"]["timing"]["median_ms"]
    del current["rows"][0]["passes"]["timing"]["timing"]["cpu_median_ms"]

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["uncomparable_rows"] == [
        {"model": "resnet18", "device": "cpu", "operation": "tl_trace"}
    ]


def test_run_timing_records_cpu_time_alongside_wall_time() -> None:
    """Timing passes record process-time statistics next to wall-clock stats."""

    stats = _run_timing(lambda: sum(range(2000)), "cpu", warmups=1, samples=5)

    assert stats["sample_count"] == 5
    assert stats["cpu_sample_count"] == 5
    assert isinstance(stats["cpu_median_ms"], float)
    assert isinstance(stats["cpu_iqr_ms"], float)
    assert len(stats["cpu_samples_ms"]) == 5


def test_run_memory_reports_phase_local_peaks() -> None:
    """Memory passes report phase-local peak deltas, not only end-state deltas."""

    payload = [torch.zeros(0)]

    def _grow() -> None:
        payload[0] = torch.ones(64, 64)

    metrics = _run_memory(_grow, "cpu", memory_runs=3)

    assert "phase_rss_high_water_delta_mb" in metrics
    assert metrics["phase_rss_high_water_delta_mb"] >= 0.0
    if metrics.get("uss_delta_mb_memory_pass") is not None:
        assert metrics["uss_peak_delta_mb_memory_pass"] >= metrics["uss_delta_mb_memory_pass"]


@pytest.mark.heavy
def test_run_memory_detects_transient_inside_measured_call() -> None:
    """An allocate-touch-free transient INSIDE ``fn()`` must register.

    r4 b6-sol R33 (HIGH) red pin: peaks were sampled only AFTER ``fn()``
    returned and the RSS fallback subtracted the process-lifetime high
    water, so after priming the process with a 256 MiB setup allocation, a
    known 128 MiB transient inside the measured call read 0.0 across every
    advertised phase-local peak field.
    """

    def _touch(buffer: bytearray) -> None:
        for index in range(0, len(buffer), 4096):
            buffer[index] = 1

    # Prime the process-lifetime high water ABOVE anything the measured
    # phase will reach, so lifetime-subtract semantics read 0.0.
    primer = bytearray(256 * 1024 * 1024)
    _touch(primer)
    del primer

    transient_mb = 128

    def _transient() -> None:
        buffer = bytearray(transient_mb * 1024 * 1024)
        _touch(buffer)
        del buffer

    metrics = _run_memory(_transient, "cpu", memory_runs=3)

    if not metrics.get("rss_high_water_phase_local"):
        pytest.skip("RSS high-water reset unavailable on this platform")
    observed = metrics["phase_rss_high_water_delta_mb"]
    assert observed >= transient_mb * 0.8, (
        f"phase-local RSS peak {observed:.1f} MB missed a {transient_mb} MB "
        "allocate-touch-free transient inside the measured call"
    )
