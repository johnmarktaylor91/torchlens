"""Hand-derived oracle for ``Module.collapse_score`` and ``Trace.module_collapse_order``.

Meaning (Lead ruling on the owner question, option B): a module's smart-collapse
priority is its hidden-op count (the distinct ops in its subtree, minus the one
op a collapsed box still shows) over the largest hidden-op count of any
submodule in the trace, rounded to 6 places. A module the renderer never
collapses reads 0.0: the root, a module with at most one Layer, and the only
child of a non-root module that hides the same ops (the parent, the name the
user wrote, ranks instead). The priority never consults the plan the
optimizer selects. Every expected value is counted from the ``forward`` sources
below.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import collapse_optimizer


class Pair(nn.Module):
    """linear, relu, linear: 3 ops, hidden 2."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class Mid(nn.Module):
    """linear, linear, relu, linear: 4 ops, hidden 3."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(torch.relu(self.b(self.a(x))))


class Big(nn.Module):
    """linear, relu, linear, tanh, linear: 5 ops, hidden 4 (the largest)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(torch.tanh(self.b(torch.relu(self.a(x)))))


class OnlyChild(nn.Module):
    """One child that hides the same 2 ops: the wrapper ranks, the child does not."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class Ranked(nn.Module):
    """big (hidden 4), mid (3), small (2), wrap over wrap.inner (2 each), head (1 Layer)."""

    def __init__(self) -> None:
        super().__init__()
        self.big = Big()
        self.mid = Mid()
        self.small = Pair()
        self.wrap = OnlyChild()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.wrap(self.small(self.mid(self.big(x)))))


# Largest submodule hidden count is big's 4: big 4/4, mid 3/4, small and
# wrap 2/4. wrap's only child hides the same 2 ops, so wrap.inner reads 0.0;
# every leaf Linear is one Layer; the root is never a candidate. Ties sort by
# address.
RANKED_ORDER = [
    ("big", 1.0),
    ("mid", 0.75),
    ("small", 0.5),
    ("wrap", 0.5),
    ("big.a", 0.0),
    ("big.b", 0.0),
    ("big.c", 0.0),
    ("head", 0.0),
    ("mid.a", 0.0),
    ("mid.b", 0.0),
    ("mid.c", 0.0),
    ("self", 0.0),
    ("small.a", 0.0),
    ("small.b", 0.0),
    ("wrap.inner", 0.0),
    ("wrap.inner.a", 0.0),
    ("wrap.inner.b", 0.0),
]


def _trace(model: nn.Module) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(model.eval(), torch.randn(2, 4))


def test_collapse_priority_matches_the_hand_ranking() -> None:
    trace = _trace(Ranked())
    try:
        assert trace.module_collapse_order == RANKED_ORDER
        scores = {module.address: module.collapse_score for module in trace.modules}
        assert scores == dict(RANKED_ORDER)
    finally:
        trace.cleanup()


def test_single_same_count_child_reads_zero_and_the_wrapper_ranks() -> None:
    # Top: w wraps w.inner (both hide linear, relu, linear: 2), then c.
    # Largest hidden count is 2, so w reads 1.0 and w.inner 0.0.
    class Top(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = OnlyChild()
            self.c = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c(self.w(x))

    trace = _trace(Top())
    try:
        scores = dict(trace.module_collapse_order)
        assert scores["w"] == 1.0
        assert scores["w.inner"] == 0.0
        assert all(score == 0.0 for address, score in scores.items() if address != "w")
    finally:
        trace.cleanup()


def test_no_collapsible_module_ranks_everything_zero() -> None:
    # Every child of the Sequential is one Layer: no candidate, all 0.0.
    trace = _trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)))
    try:
        assert trace.module_collapse_order == [
            ("0", 0.0),
            ("1", 0.0),
            ("2", 0.0),
            ("self", 0.0),
        ]
    finally:
        trace.cleanup()


def test_past_the_pathological_pre_gate_the_ranking_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace(Ranked())
    try:
        monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)
        assert trace.module_collapse_order == []
        assert trace.modules["big"].collapse_score == 0.0
    finally:
        trace.cleanup()
