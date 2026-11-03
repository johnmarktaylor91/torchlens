"""Observe-kit items 2-3: execution-order compare() + the isolated-rerun harness.

Pins the two shipped-surface repairs the observe memo sequenced FIRST:
``compare()`` rows follow execution order (never alphabetical), and the
shared double-run harness releases each fresh model copy so double-run tools
work on an already-traced model (the deepcopy-after-trace ``KeyError``).
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.debug import (
    bisect_precision,
    first_divergence,
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


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_graph.py.
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


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_graph.py.
def test_first_divergence_none_when_identical() -> None:
    """Identical runs return None instead of a spurious row."""

    x = torch.linspace(-1.0, 1.0, 8)
    trace_a = tl.trace(_TanhThenAdd(0.5), x)
    trace_b = tl.trace(_TanhThenAdd(0.5), x)
    try:
        assert first_divergence(trace_a, trace_b) is None
    finally:
        _cleanup(trace_a, trace_b)


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_rerun.py.
def test_clone_input_tree_copies_tensors_and_containers() -> None:
    """The clone shares no tensor or container with the input tree."""

    weight = torch.randn(3, requires_grad=True)
    tag = object()
    tree = {"a": [torch.ones(2), (weight, 4)], "b": tag}

    cloned = tl.debug.clone_input_tree(tree)

    assert cloned is not tree and cloned["a"] is not tree["a"]
    assert cloned["a"][1] is not tree["a"][1]
    assert isinstance(cloned["a"], list) and isinstance(cloned["a"][1], tuple)
    assert cloned["b"] is tag and cloned["a"][1][1] == 4
    cloned_weight = cloned["a"][1][0]
    assert cloned_weight is not weight
    assert cloned_weight.data_ptr() != weight.data_ptr()
    assert torch.equal(cloned_weight, weight)
    assert cloned_weight.requires_grad and cloned_weight.grad_fn is None
    cloned["a"][0].add_(5.0)
    cloned["a"].append("extra")
    assert torch.equal(tree["a"][0], torch.ones(2))
    assert len(tree["a"]) == 2


# Kept: regression test for a fixed bug; the shrink-r4 regression class needs a planted fault to delete it.
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


# Kept: its run phase reached no library code in the per-id coverage run, so it has no measurement (shrink rule 3).
def test_rerun_helpers_are_public_in_tl_debug() -> None:
    """``clone_input_tree`` and ``preserved_rng_state`` are ``tl.debug`` exports."""

    from torchlens.debug import _rerun

    for name in ("clone_input_tree", "preserved_rng_state"):
        assert name in tl.debug.__all__
        assert getattr(tl.debug, name) is getattr(_rerun, name)
