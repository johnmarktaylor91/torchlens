"""Collapse eligibility gate: a single child that hides the same ops.

A module whose only child hides exactly its own ops is never a collapse
candidate (collapsing it shows the same box as collapsing the child). Signals
are built parent-first, so the gate must not depend on the child's signal
having been built already.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.auto_collapse import analyze_collapse


class _Pair(nn.Module):
    """linear, relu, linear: 3 ops, hidden 2."""

    def __init__(self) -> None:
        """Initialize two linears."""
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear, relu, linear."""
        return self.b(torch.relu(self.a(x)))


class _OnlyChild(nn.Module):
    """One child that hides the same 2 ops."""

    def __init__(self) -> None:
        """Initialize the single child."""
        super().__init__()
        self.inner = _Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the child."""
        return self.inner(x)


class _ChildPlusOp(nn.Module):
    """One child (hidden 2) plus an own tanh: hidden 3, so it stays a candidate."""

    def __init__(self) -> None:
        """Initialize the single child."""
        super().__init__()
        self.inner = _Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child, then tanh."""
        return torch.tanh(self.inner(x))


class _Top(nn.Module):
    """w wraps w.inner (same hidden ops), v wraps v.inner plus one op, then c."""

    def __init__(self) -> None:
        """Initialize the wrappers and a head."""
        super().__init__()
        self.w = _OnlyChild()
        self.v = _ChildPlusOp()
        self.c = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run w, v, then the head."""
        return self.c(self.v(self.w(x)))


def test_single_same_hidden_child_parent_is_not_eligible() -> None:
    """The parent of a same-hidden single child is gated off; the child stays eligible."""
    torch.manual_seed(0)
    trace = tl.trace(_Top().eval(), torch.randn(2, 4))
    try:
        signals = analyze_collapse(trace).signals
        assert signals["w"].hidden_ops == signals["w.inner"].hidden_ops == 2
        assert signals["w"].eligible is False
        assert signals["w.inner"].eligible is True
        # A single child that hides fewer ops than its parent does not gate the parent.
        assert signals["v"].hidden_ops == 3
        assert signals["v.inner"].hidden_ops == 2
        assert signals["v"].eligible is True
        assert signals["v.inner"].eligible is True
        # The root and one-Layer leaves stay gated as before.
        assert signals["self"].eligible is False
        assert signals["c"].eligible is False
    finally:
        trace.cleanup()
