"""Collapse eligibility gate: a single child that hides the same ops.

When a non-root module's only child hides exactly the parent's ops, collapsing
either shows the same box. The parent, the name the user wrote, is the box
candidate and the child is not; in a chain of such wrappers the outermost stays
the candidate. A child that hides fewer ops, a parent with two children, and
the root's only child keep their own eligibility. Signals are built
parent-first, so the gate must not depend on build order.
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
    """One child that hides the same ops as this wrapper."""

    def __init__(self, inner: nn.Module | None = None) -> None:
        """Initialize the single child (a ``_Pair`` by default)."""
        super().__init__()
        self.inner = _Pair() if inner is None else inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the child."""
        return self.inner(x)


class _ChildPlusOp(nn.Module):
    """One child (hidden 2) plus an own tanh: hidden 3, so nothing is gated."""

    def __init__(self) -> None:
        """Initialize the single child."""
        super().__init__()
        self.inner = _Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child, then tanh."""
        return torch.tanh(self.inner(x))


class _TwoChildren(nn.Module):
    """Two children: a pair (hidden 2) and a linear; hidden 3."""

    def __init__(self) -> None:
        """Initialize both children."""
        super().__init__()
        self.inner = _Pair()
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pair, then the linear."""
        return self.head(self.inner(x))


class _Top(nn.Module):
    """w wraps w.inner (same ops); v adds one op; t has two children; then c."""

    def __init__(self) -> None:
        """Initialize the wrappers and a head."""
        super().__init__()
        self.w = _OnlyChild()
        self.v = _ChildPlusOp()
        self.t = _TwoChildren()
        self.c = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run w, v, t, then the head."""
        return self.c(self.t(self.v(self.w(x))))


def _signals(model: nn.Module) -> dict[str, tuple[int, bool]]:
    """Return ``{address: (hidden_ops, eligible)}`` for one capture."""
    torch.manual_seed(0)
    trace = tl.trace(model.eval(), torch.randn(2, 4))
    try:
        signals = analyze_collapse(trace).signals
        return {address: (s.hidden_ops, s.eligible) for address, s in signals.items()}
    finally:
        trace.cleanup()


def test_single_same_hidden_child_is_gated_and_the_parent_stays_eligible() -> None:
    """The only child that hides its parent's ops is gated off; the parent keeps the box."""
    signals = _signals(_Top())
    assert signals["w"] == (2, True)
    assert signals["w.inner"] == (2, False)
    # The root and one-Layer leaves stay gated as before.
    assert signals["self"][1] is False
    assert signals["c"][1] is False


def test_child_hiding_fewer_ops_and_two_child_parent_keep_their_eligibility() -> None:
    """Armed negative: the gate needs one child that hides exactly the parent's ops."""
    signals = _signals(_Top())
    assert signals["v"] == (3, True)
    assert signals["v.inner"] == (2, True)
    assert signals["t"] == (3, True)
    assert signals["t.inner"] == (2, True)
    assert signals["t.head"][1] is False  # one Layer


def test_wrapper_chain_keeps_the_outermost_and_the_root_never_gates_its_child() -> None:
    """In a chain of same-op wrappers only the outermost non-root module is a candidate."""
    signals = _signals(_OnlyChild(_OnlyChild(_OnlyChild())))
    assert signals["self"] == (2, False)
    # The root is never a candidate, so its only child is not gated by it.
    assert signals["inner"] == (2, True)
    assert signals["inner.inner"] == (2, False)
    assert signals["inner.inner.inner"] == (2, False)
