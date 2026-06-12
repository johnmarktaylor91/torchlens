"""Tests for P6 benchmark gate and generated performance snippets."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from benchmarks.generate_perf_numbers import render_numbers_markdown
from benchmarks.perf_gate import compare_gate_payloads, normalize_gate_payload


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
            "date": "2026-06-12",
            "environment": {"torchlens_git_sha": "abc1234"},
            "rows": rows,
        }
    )


def test_perf_gate_compare_passes_within_tolerance() -> None:
    """Regression gate accepts rows inside the P6 tolerance."""

    baseline = _payload([_row("resnet18", "cpu", "tl_trace", 100.0, iqr_ms=3.0)])
    current = _payload([_row("resnet18", "cpu", "tl_trace", 111.0, iqr_ms=6.0)])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is True
    assert comparison["checks"][0]["tolerance_ms"] == 12.0


def test_perf_gate_compare_fails_regression_beyond_tolerance() -> None:
    """Regression gate rejects median slowdowns beyond tolerance."""

    baseline = _payload([_row("resnet18", "cpu", "tl_trace", 100.0, iqr_ms=1.0)])
    current = _payload([_row("resnet18", "cpu", "tl_trace", 130.0, iqr_ms=1.0)])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["regressions"][0]["delta_ms"] == 30.0


def test_perf_gate_requires_current_torchlens_rows_ok() -> None:
    """Regression gate rejects skipped or errored current TorchLens rows."""

    baseline = _payload([_row("resnet18", "cpu", "fastlog_halt_25", 10.0)])
    current = _payload([_row("resnet18", "cpu", "fastlog_halt_25", 9.0, status="skipped")])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["status_failures"] == [
        {"model": "resnet18", "device": "cpu", "operation": "fastlog_halt_25", "status": "skipped"}
    ]


def test_generated_numbers_print_halt_headline_only_when_sub_raw() -> None:
    """Generator prints the fractional-x halt headline only when measured sub-1.0x."""

    faster_payload = _payload(
        [
            _row("resnet18", "cpu", "raw_forward", 20.0),
            _row("resnet18", "cpu", "fastlog_halt_25", 18.0),
        ]
    )
    slower_payload = _payload(
        [
            _row("resnet18", "cpu", "raw_forward", 20.0),
            _row("resnet18", "cpu", "fastlog_halt_25", 21.0),
        ]
    )

    faster = render_numbers_markdown(faster_payload)
    slower = render_numbers_markdown(slower_payload)

    assert "Headline: measured `fastlog_halt_25` is 0.90x raw forward" in faster
    assert "Headline: measured `fastlog_halt_25`" not in slower
    assert "| resnet18 | cpu | fastlog_halt_25 | 21.0 | 1.05x | ok |" in slower


def test_emit_tensor_grad_event_populates_profile_buckets() -> None:
    """Tensor grad hooks record P6 profiling buckets for later optimization."""

    model = nn.Linear(3, 2)
    x = torch.randn(1, 3, requires_grad=True)
    trace = tl.trace(model, x, backward_ready=True, profile=True)

    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    timings = trace._phase_timings
    assert timings["backward_grad_event:ensure_stream"]["count"] >= 1
    assert timings["backward_grad_event:ensure_pass"]["count"] >= 1
    assert timings["backward_grad_event:payload"]["count"] >= 1
    assert timings["backward_grad_event:append"]["count"] >= 1
