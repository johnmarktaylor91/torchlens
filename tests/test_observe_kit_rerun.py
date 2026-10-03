"""Observe-kit items 2-3: execution-order compare() + the isolated-rerun harness.

Pins the two shipped-surface repairs the observe memo sequenced FIRST:
``compare()`` rows follow execution order (never alphabetical), and the
shared double-run harness releases each fresh model copy so double-run tools
work on an already-traced model (the deepcopy-after-trace ``KeyError``).
"""

from __future__ import annotations

import random

import torch
from torch import nn

import torchlens as tl
from torchlens.debug import (
    bisect_precision,
    compare_rows,
    first_divergence,
    isolated_capture,
    preserved_rng_state,
)


class _TanhThenAdd(nn.Module):
    """Two-op model whose execution order is the REVERSE of sorted() order."""

    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``tanh`` (executes first, sorts last) then ``add``."""

        y = torch.tanh(x)
        return y + self.constant


class _MulTanhAdd(nn.Module):
    """Three-op model where the FIRST executed op diverges but sorts last."""

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``mul`` -> ``tanh`` -> ``add``; a weight change diverges all three."""

        y = torch.tanh(x * self.weight)
        return y + 1.0


def _cleanup(*traces: object) -> None:
    """Release captured traces so the process-global registry stays clean."""

    for captured in traces:
        cleanup = getattr(captured, "cleanup", None)
        if cleanup is not None:
            cleanup()


def test_compare_rows_follow_execution_order() -> None:
    """Rows walk execution order even when sorted() order disagrees."""

    x = torch.linspace(-1.0, 1.0, 8)
    trace_a = tl.trace(_TanhThenAdd(0.5), x)
    trace_b = tl.trace(_TanhThenAdd(0.75), x)
    try:
        rows, attrs = compare_rows(trace_a, trace_b)
        labels = [row["op"] for row in rows]
        tanh_position = next(i for i, label in enumerate(labels) if label.startswith("tanh"))
        add_position = next(i for i, label in enumerate(labels) if label.startswith("add"))
        # Alphabetical order would put add_* first; execution order is tanh -> add.
        assert tanh_position < add_position
        assert attrs["value_diverged"] >= 1
    finally:
        _cleanup(trace_a, trace_b)


def test_first_divergence_names_first_executed_divergence() -> None:
    """The first-diverging-op recipe names the op that actually diverged first.

    Every compute op diverges when the weight changes; the alphabetical walk
    named ``add_*`` (sorts first) while the true first divergence in execution
    order is the ``mul`` at step 1.
    """

    x = torch.linspace(-1.0, 1.0, 8)
    trace_a = tl.trace(_MulTanhAdd(1.0), x)
    trace_b = tl.trace(_MulTanhAdd(2.0), x)
    try:
        row = first_divergence(trace_a, trace_b)
        assert row is not None
        assert row["op"].startswith("mul"), row
        assert row["allclose"] is False
    finally:
        _cleanup(trace_a, trace_b)


def test_first_divergence_none_when_identical() -> None:
    """Identical runs return None instead of a spurious row."""

    x = torch.linspace(-1.0, 1.0, 8)
    trace_a = tl.trace(_TanhThenAdd(0.5), x)
    trace_b = tl.trace(_TanhThenAdd(0.5), x)
    try:
        assert first_divergence(trace_a, trace_b) is None
    finally:
        _cleanup(trace_a, trace_b)


def test_bisect_precision_works_after_prior_trace() -> None:
    """The double-run harness survives an already-traced source model.

    ``copy.deepcopy`` of a traced model used to carry stale instance-level
    forward wrappers into the copies and the next capture died with a bare
    ``KeyError``; the shared harness releases each copy first.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)
    prior = tl.trace(model, x)
    try:
        result = bisect_precision(model, x)
        assert result.rows, "expected compared rows from an already-traced model"
    finally:
        _cleanup(prior)


def test_isolated_capture_preserves_caller_rng_and_reproduces() -> None:
    """Same-seed isolated captures agree; caller RNG streams are untouched."""

    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
    model.train()
    x = torch.randn(2, 4)
    torch.manual_seed(1234)
    random.seed(99)
    torch_state_before = torch.get_rng_state()
    python_state_before = random.getstate()

    run_a = isolated_capture(model, x, seed=7)
    run_b = isolated_capture(model, x, seed=7)
    run_c = isolated_capture(model, x, seed=8)
    try:
        assert torch.equal(torch.get_rng_state(), torch_state_before)
        assert random.getstate() == python_state_before
        out_a = run_a.output_ops[0].out
        out_b = run_b.output_ops[0].out
        out_c = run_c.output_ops[0].out
        assert torch.equal(out_a, out_b)
        assert not torch.equal(out_a, out_c)
    finally:
        _cleanup(run_a, run_b, run_c)


def test_preserved_rng_state_round_trips() -> None:
    """Draws inside the preserved scope are invisible to the caller."""

    torch.manual_seed(5)
    random.seed(5)
    expected_torch = torch.get_rng_state()
    expected_python = random.getstate()
    with preserved_rng_state():
        torch.randn(16)
        random.random()
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert random.getstate() == expected_python
