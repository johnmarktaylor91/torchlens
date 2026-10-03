"""Behavioral tests for debug cost ranking, byte model, compile counting, and audits."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402
from torchlens.debug import audit_trace, count_compiles, hot_path  # noqa: E402
from torchlens.debug._cost import theoretical_op_bytes  # noqa: E402


class _TwoLineModel(nn.Module):
    """Model whose forward spreads ops over distinct source lines."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        z = torch.relu(y)
        return z.reshape(-1)


@pytest.fixture(scope="module")
def cost_trace() -> Iterator[tl.Trace]:
    """One captured trace shared across cost assertions."""

    torch.manual_seed(12)
    trace = tl.trace(_TwoLineModel(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_hot_path_ranks_source_lines_with_full_percentages(cost_trace) -> None:
    """hot_path() groups ops by source line, sorts by cost, and sums to 100%."""

    frame = hot_path(cost_trace, by="duration")
    assert list(frame.columns) == ["source_file:line", "op_count", "total_cost", "pct_total"]
    assert len(frame) >= 2
    costs = list(frame["total_cost"])
    assert costs == sorted(costs, reverse=True)
    assert sum(frame["pct_total"]) == pytest.approx(100.0)
    assert int(frame["op_count"].sum()) >= 3
    assert frame.attrs["metric"] == "duration"
    assert "test_debug_cost_audit_behaviors" in str(frame["source_file:line"].iloc[0])


def test_hot_path_rejects_unknown_metric(cost_trace) -> None:
    """An unsupported cost metric fails fast."""

    with pytest.raises(KeyError):
        hot_path(cost_trace, by="watts")


@pytest.mark.smoke
def test_theoretical_op_bytes_models_reads_writes_and_views(cost_trace) -> None:
    """Dense ops read inputs+params once and write once; views cost zero."""

    linear = cost_trace["linear_1_1"]
    bytes_read, bytes_written = theoretical_op_bytes(linear)
    # input (2,4) fp32 + weight (4,4) fp32 + bias (4,) fp32
    assert int(bytes_read) == (2 * 4 + 4 * 4 + 4) * 4
    assert int(bytes_written) == 2 * 4 * 4

    view = cost_trace["reshape_1_3"]
    view_read, view_written = theoretical_op_bytes(view)
    assert int(view_read) == 0
    assert int(view_written) == 0


@pytest.mark.smoke
def test_count_compiles_reports_zero_for_eager_blocks() -> None:
    """A block with no Dynamo activity reports zero frame compilations."""

    with count_compiles() as counts:
        _ = torch.relu(torch.ones(3))
    assert counts.frames_compiled == 0


@pytest.mark.slow
def test_count_compiles_sees_live_compiles_and_freezes_at_exit() -> None:
    """Compiles inside a block are counted; exited blocks never absorb later ones."""

    with count_compiles() as before:
        pass

    with count_compiles() as during:
        compiled = torch.compile(lambda x: x * 2 + 1)
        compiled(torch.ones(3))
        live_count = during.frames_compiled
    assert live_count >= 1
    assert during.frames_compiled == live_count

    # The earlier block exited before the compile; its frozen count must not
    # have absorbed the later event.
    assert before.frames_compiled == 0


class _MidwayFailModel(nn.Module):
    """Model that fails after producing a non-finite intermediate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(x)
        _ = y / 0.0
        raise RuntimeError("midway boom")


@pytest.mark.smoke
def test_audit_trace_on_failed_partial_reports_exception_with_context() -> None:
    """A failed capture audits to a critical finding naming the exception."""

    with pytest.raises(RuntimeError, match="midway boom") as excinfo:
        tl.trace(_MidwayFailModel(), torch.ones(2, 3))

    partial = tl.partial.from_failed_capture(excinfo.value)
    audit = audit_trace(partial)

    by_check = {finding.check: finding for finding in audit.findings}
    exception_finding = by_check["partial_capture_exception"]
    assert exception_finding.severity == "critical"
    assert "RuntimeError" in exception_finding.message
    assert "midway boom" in exception_finding.message
    # The division by zero before the raise is surfaced as a non-finite finding.
    assert "find_nan" in by_check
    assert audit.skipped  # partial audits disclose unrun full-trace checks
    text = repr(audit)
    assert "issue(s)" in text and "skipped" in text
