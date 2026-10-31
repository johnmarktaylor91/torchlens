"""Guarded fast live run: site collection follows the capture on same-input reruns.

Each model here reproduces one refusal the behaviour net found on a same-input
fast rerun of a deterministic model, where the full run succeeds.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens._fast_run import _FastLiveSession


def _fast_matches_full(model: nn.Module, inputs: torch.Tensor) -> None:
    """Capture, then assert the fast rerun on the same input returns the captured output."""

    torch.manual_seed(0)
    trace = tl.trace(model, inputs.clone())
    expected = [op.out for op in trace.output_ops]
    torch.manual_seed(0)
    result = trace.run(inputs=inputs.clone(), fast=True)
    got = [result.output] if isinstance(result.output, torch.Tensor) else list(result.output)
    assert len(got) == len(expected)
    for want, have in zip(expected, got):
        assert torch.equal(want, have.detach())


class StrayStatementPow(nn.Module):
    """Two consecutive ``pow`` calls whose first result dies at once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Free one ``pow`` result before computing the next."""

        squared = x.pow(2)
        del squared
        return x.pow(2)


def _reused_tensor_address(first: torch.Tensor) -> torch.Tensor | None:
    """Free ``first`` and allocate until a tensor takes its address (CPython reuses freed slots)."""

    wanted = id(first)
    del first
    for _ in range(64):
        candidate = torch.ones(3)
        if id(candidate) == wanted:
            return candidate
    return None


def test_consecutive_same_name_events_are_told_apart_by_object_identity() -> None:
    """A fresh output at a freed tensor's address is a new event, not a nested repeat."""

    model = StrayStatementPow()
    trace = tl.trace(model, torch.ones(3))
    session = _FastLiveSession(trace, model)
    try:
        with session.activated():
            first = torch.ones(3)
            session.capture_function("pow", first)
            assert session.function_index == 1
            second = _reused_tensor_address(first)
            del first
            if second is None:
                pytest.skip("the allocator did not reuse the freed tensor's address")
            session.capture_function("pow", second)
            assert session.failure is None
            assert session.function_index == 2
    finally:
        session.close()


def test_fast_run_consecutive_same_name_ops_match_full_run() -> None:
    """End to end: two consecutive same-name ops run fast and match the capture."""

    _fast_matches_full(StrayStatementPow(), torch.arange(6.0).view(2, 3))


class UnpackedPool(nn.Module):
    """``F.adaptive_max_pool1d`` unpacks the tuple of its nested ``_with_indices`` call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool, then view: the view must follow one pooling event, not two."""

        pooled = F.adaptive_max_pool1d(x, 2)
        return pooled.view(pooled.shape[0], -1)


def test_fast_run_nested_same_name_call_returning_an_inner_leaf_is_one_event() -> None:
    """The outer call's tensor is a leaf of the inner tuple: one captured op, one event."""

    _fast_matches_full(UnpackedPool(), torch.arange(16.0).view(2, 2, 4))


class SetItemSite(nn.Module):
    """A ``__setitem__`` on an activation: Python returns ``None`` for it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate one row in place and keep computing from the receiver."""

        y = x * 2
        y[0] = 1
        return y + 1


def test_fast_run_setitem_site_collects_the_mutated_receiver() -> None:
    """Capture records the receiver as the setter's output; the fast run sees the same."""

    _fast_matches_full(SetItemSite(), torch.arange(6.0).view(2, 3))


class PairChild(nn.Module):
    """Returns a tuple whose second leaf is computed after the first."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a, b)`` with ``b`` the later op."""

        a = x + 1
        b = a * 2
        return a, b


class EarlyLeafParent(nn.Module):
    """Returns the child's EARLY leaf, so its last output op precedes the child's."""

    def __init__(self) -> None:
        super().__init__()
        self.child = PairChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hand back the first leaf of the child's pair."""

        a, _b = self.child(x)
        return a


class EarlyLeafModel(nn.Module):
    """Root holding the parent, so the parent is a planned atomic exit."""

    def __init__(self) -> None:
        super().__init__()
        self.parent = EarlyLeafParent()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the parent then one op."""

        return self.parent(x) - 1


def test_fast_run_module_plans_follow_module_exit_order() -> None:
    """A parent returning a child's early leaf still exits after that child."""

    _fast_matches_full(EarlyLeafModel(), torch.arange(4.0))
