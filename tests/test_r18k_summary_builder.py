"""Regression tests for r18k summary-builder hardening (group K).

Covers the five group-K findings in
``torchlens/visualization/_summary_internal/_builder.py``:

* H2  -- ``summary(mode="rolled")`` / ``show_ops=True`` must not raise the
  multi-pass accessor ``ValueError`` on recurrent models.
* H8  -- the graph/overview "Connected To" column must report REAL dataflow
  topology, never fabricated containment-tree parents.
* control_flow -- must not deny recurrence on a recurrent trace.
* xN  -- unrolled per-pass rows must carry pass-qualified names, not the
  aggregate ``xN`` multiplicity.
* F8  -- the compute footer must not pass capture wall time off as forward
  compute cost; it must agree with the waterfall level.

These are honesty tripwires: each assertion fails against the pre-fix source.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torchlens as tl


class _Reused(nn.Module):
    """Recurrent: one ReLU replayed three times (rolls to a 3-pass layer)."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.relu(x)
        return x


class _TwoMod(nn.Module):
    """Sequential: ``b`` consumes ``a``'s output."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _Branch(nn.Module):
    """Branching: ``c`` consumes ``a(x) + b(x)`` (a bare add op)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(self.a(x) + self.b(x))


def _recurrent_trace() -> tl.Trace:
    return tl.trace(_Reused(), torch.ones(1024))


# --------------------------------------------------------------------------- H2


def test_h2_rolled_summary_no_crash_on_recurrent() -> None:
    """Rolled waterfall + rolled show_ops at every level must not raise."""
    trace = _recurrent_trace()
    assert trace.is_recurrent
    # Previously raised ValueError from Layer.func_duration.
    trace.summary(level="waterfall", mode="rolled")
    for level in ("overview", "graph", "memory", "compute", "waterfall"):
        trace.summary(level=level, mode="rolled", show_ops=True)


def test_h2_rolled_time_is_honest_aggregate() -> None:
    """The rolled recurrent row must report the aggregate (summed) time."""
    trace = _recurrent_trace()
    rolled = trace.summary(level="waterfall", mode="rolled")
    unrolled = trace.summary(level="waterfall", mode="unrolled")

    def _accumulated(text: str) -> float:
        for line in text.splitlines():
            if "Accumulated op time:" in line:
                return float(line.split(":")[1].strip().split()[0])
        raise AssertionError("no accumulated op time footer")

    # Rolled aggregate must match the sum of the unrolled per-pass durations.
    assert _accumulated(rolled) == _accumulated(unrolled)


# --------------------------------------------------------------------------- H8


def _connected_to(trace: tl.Trace, module_address: str) -> str:
    row = None
    for line in trace.summary(level="graph").splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells and cells[0].startswith(f"{module_address} ("):
            row = cells
    assert row is not None, f"no graph row for module {module_address!r}"
    return row[-1]


def test_h8_sequential_topology_is_real_dataflow() -> None:
    """A chain a -> b must not report both modules connected to 'input'."""
    trace = tl.trace(_TwoMod(), torch.randn(2, 8))
    assert _connected_to(trace, "a") == "input"
    # The critical assertion: b is fed by a, NOT the fabricated 'input'.
    assert _connected_to(trace, "b") == "a"


def test_h8_branch_topology_names_real_producer() -> None:
    trace = tl.trace(_Branch(), torch.randn(2, 8))
    assert _connected_to(trace, "a") == "input"
    assert _connected_to(trace, "b") == "input"
    # c is fed by the elementwise add, not 'input'.
    assert _connected_to(trace, "c") != "input"
    assert "add" in _connected_to(trace, "c")


# ------------------------------------------------------------------ control_flow


def test_control_flow_discloses_recurrence() -> None:
    trace = _recurrent_trace()
    text = trace.summary(level="control_flow")
    # Must NOT deny recurrent loop groups on a recurrent trace.
    assert "recurrent loop groups were detected" not in text
    assert "Recurrent loop groups" in text
    assert "relu_1_1" in text


def test_control_flow_honest_when_no_recurrence() -> None:
    trace = tl.trace(nn.Linear(4, 4), torch.ones(1, 4))
    assert not trace.is_recurrent
    text = trace.summary(level="control_flow")
    # Non-recurrent feedforward keeps the original honest empty-state message.
    assert "No conditional branches or recurrent loop groups were detected" in text


# --------------------------------------------------------------------------- xN


def _memory_row_names(trace: tl.Trace, mode: str, needle: str) -> list[str]:
    names = []
    for line in trace.summary(level="memory", mode=mode).splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells and needle in cells[0]:
            names.append(cells[0])
    return names


def test_xn_unrolled_rows_are_pass_qualified() -> None:
    trace = _recurrent_trace()
    names = _memory_row_names(trace, "unrolled", "relu_1_1")
    assert names == ["relu_1_1:1", "relu_1_1:2", "relu_1_1:3"]


def test_xn_rolled_row_keeps_aggregate_multiplicity() -> None:
    trace = _recurrent_trace()
    names = _memory_row_names(trace, "rolled", "relu_1_1")
    # Rolled aggregate stays a single row; the multiplicity is spelled in the
    # one pass/op vocabulary (A10): "xN passes", never bare "xN" or "N ops".
    assert names == ["relu_1_1 (x3 passes)"]


# --------------------------------------------------------------------------- F8


def _footer_value(text: str, needle: str) -> float:
    for line in text.splitlines():
        if needle in line:
            return float(line.split(":")[-1].strip().split()[0])
    raise AssertionError(f"footer line {needle!r} not found")


def test_f8_compute_time_agrees_with_waterfall() -> None:
    trace = _recurrent_trace()
    compute = trace.summary(level="compute")
    waterfall = trace.summary(level="waterfall", mode="unrolled")
    # No more mislabeled 'Forward time' that reads as compute cost.
    assert "Forward time:" not in compute
    # Headline compute time now equals the waterfall accumulated op time.
    assert _footer_value(compute, "Accumulated op time:") == _footer_value(
        waterfall, "Accumulated op time:"
    )
    # The overhead-inclusive wall time is disclosed as such, not as forward cost.
    assert "Capture wall time (includes TorchLens overhead):" in compute
