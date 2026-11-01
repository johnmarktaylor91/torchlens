"""Guarded fast live run: site collection follows the capture on same-input reruns.

Each model here reproduces one refusal the behaviour net found on a same-input
fast rerun of a deterministic model, where the full run succeeds.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

import torchlens as tl
from torchlens._fast_run import _FastLiveSession
from torchlens.backends.torch._tl import get_module_meta
from torchlens.errors import RunCapabilityUnavailableError


def _fast_matches_full(model: nn.Module, inputs: torch.Tensor) -> None:
    """Capture, then assert the fast rerun on the same input returns the captured output."""

    torch.manual_seed(0)
    trace = tl.trace(model, inputs.clone())
    expected = [op.out for op in trace.output_ops]
    torch.manual_seed(0)
    result = trace.run(inputs=inputs.clone(), fast=True)
    got = [result.output] if isinstance(result.output, torch.Tensor) else list(result.output)
    assert len(got) == len(expected)
    for want, have in zip(expected, got, strict=True):
        assert torch.equal(want, have.detach())


class StrayStatementPow(nn.Module):
    """Two consecutive ``pow`` calls whose first result dies at once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Free one ``pow`` result before computing the next."""

        squared = x.pow(2)
        del squared
        return x.pow(2)


def _tensor_preferring_address(wanted: int) -> torch.Tensor:
    """Allocate tensors until one takes the address ``wanted`` (CPython reuses freed slots), else the last."""

    candidate = torch.ones(3)
    for _ in range(64):
        if id(candidate) == wanted:
            break
        candidate = torch.ones(3)
    return candidate


def test_consecutive_same_name_events_are_told_apart_by_object_identity() -> None:
    """A fresh output after the last one is freed is a new event, not a nested repeat.

    Before the fix the collector compared ``id()`` values, so the fresh tensor
    that took the freed output's address was dropped as a repeat.
    """

    model = StrayStatementPow()
    trace = tl.trace(model, torch.ones(3))
    session = _FastLiveSession(trace, model)
    try:
        with session.activated():
            first = torch.ones(3)
            wanted = id(first)
            session.capture_function("pow", first)
            assert session.function_index == 1
            del first
            second = _tensor_preferring_address(wanted)
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


class AliasedEmbedding(nn.Module):
    """``self.B`` is the same module object as ``self.A[0]``."""

    def __init__(self) -> None:
        super().__init__()
        self.A = nn.ModuleList([nn.Linear(3, 3)])
        self.B = self.A[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the shared module through its alias."""

        return self.B(x) * 2


def test_fast_run_resolves_alias_module_addresses() -> None:
    """A module capture labels by an alias address is found on the live model."""

    _fast_matches_full(AliasedEmbedding(), torch.arange(6.0).view(2, 3))


class DirectForwardParent(nn.Module):
    """Calls its child's ``forward`` directly, bypassing torch module hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reach the child without ``__call__``."""

        return self.inner.forward(x)


class DirectForwardModel(nn.Module):
    """Root holding the parent whose child is reached by a direct forward() call."""

    def __init__(self) -> None:
        super().__init__()
        self.parent = DirectForwardParent()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the parent then one op."""

        return self.parent(x) + 1


def test_fast_run_names_an_unrecorded_direct_forward_call_as_a_capability_refusal() -> None:
    """A direct forward() call with no capture record is refused as unobservable.

    Capture marks the modules it entered by a direct call and the fast run
    routes those; without the mark (here removed by hand, as on a torch where
    ``HAS_MODULE_CALL_DISPATCH_FRAME`` is False) the call stays hook-invisible
    and is refused as a capability gap, not as a path change.
    """

    model = DirectForwardModel()
    trace = tl.trace(model, torch.arange(6.0).view(2, 3))
    meta = get_module_meta(model.parent.inner)
    assert meta is not None and meta.direct_forward_called
    meta.direct_forward_called = False
    with pytest.raises(RunCapabilityUnavailableError) as info:
        trace.run(inputs=torch.arange(6.0).view(2, 3), fast=True)
    assert info.value.fields["code"] == "run_capability_unavailable"
    assert "forward()" in str(info.value)
    assert "parent.inner" in str(info.value)
    assert "recorded no such call" in str(info.value)


class DistributionOutput(nn.Module):
    """Returns a ``torch.distributions.Normal``, an opaque output container."""

    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Linear(3, 2)
        self.log_std = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> Normal:
        """Build the distribution from two heads."""

        return Normal(self.mu(x), self.log_std(x).clamp(-5, 2).exp())


def test_fast_run_names_an_opaque_output_container_as_a_capability_refusal() -> None:
    """A collapsed opaque boundary is refused as unpairable, not as a changed structure."""

    model = DistributionOutput()
    # Capture's own advisory for an unsupported output container (the BFS
    # fallback) is the documented behaviour here, not the subject.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace = tl.trace(model, torch.arange(6.0).view(2, 3))
        with pytest.raises(RunCapabilityUnavailableError) as info:
            trace.run(inputs=torch.arange(6.0).view(2, 3), fast=True)
    assert info.value.fields["code"] == "run_capability_unavailable"
    assert "Normal" in str(info.value)


class ClonesItsActivation(nn.Module):
    """Clones an activation, so ``clone`` is a planned functional site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply, clone, add."""

        y = x * 2
        z = y.clone()
        return z + 1


def test_fast_run_ignores_its_own_payload_clone() -> None:
    """The session's payload copy must not be counted as the model's ``clone`` event."""

    _fast_matches_full(ClonesItsActivation(), torch.arange(6.0).view(2, 3))
