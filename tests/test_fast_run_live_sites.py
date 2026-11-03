"""Guarded fast live run: site collection follows the capture on same-input reruns.

Each model here reproduces one refusal the behaviour net found on a same-input
fast rerun of a deterministic model, where the full run succeeds.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F

import torchlens as tl
from torchlens.backends.torch._tl import get_module_meta
from torchlens.errors import RunCapabilityUnavailableError


class StrayStatementPow(nn.Module):
    """Two consecutive ``pow`` calls whose first result dies at once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Free one ``pow`` result before computing the next."""

        squared = x.pow(2)
        del squared
        return x.pow(2)


class UnpackedPool(nn.Module):
    """``F.adaptive_max_pool1d`` unpacks the tuple of its nested ``_with_indices`` call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool, then view: the view must follow one pooling event, not two."""

        pooled = F.adaptive_max_pool1d(x, 2)
        return pooled.view(pooled.shape[0], -1)


class SetItemSite(nn.Module):
    """A ``__setitem__`` on an activation: Python returns ``None`` for it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate one row in place and keep computing from the receiver."""

        y = x * 2
        y[0] = 1
        return y + 1


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


class AliasedEmbedding(nn.Module):
    """``self.B`` is the same module object as ``self.A[0]``."""

    def __init__(self) -> None:
        super().__init__()
        self.A = nn.ModuleList([nn.Linear(3, 3)])
        self.B = self.A[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the shared module through its alias."""

        return self.B(x) * 2


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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _fast_live_plans.py, _fast_run.py.
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
    """Returns a ``torch.distributions.Normal``: a structured output keyed by parameter name."""

    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Linear(3, 2)
        self.log_std = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> Normal:
        """Build the distribution from two heads."""

        return Normal(self.mu(x), self.log_std(x).clamp(-5, 2).exp())


class SquashedDistributionOutput(DistributionOutput):
    """Returns a tanh-squashed Normal: its transform list is state capture cannot record."""

    def forward(self, x: torch.Tensor) -> TransformedDistribution:  # type: ignore[override]
        """Squash the parent's Normal."""

        return TransformedDistribution(super().forward(x), [TanhTransform()])


class _Holder:
    """A plain class holding tensors: an output capture walks only by attribute search."""

    def __init__(self, first: torch.Tensor, second: torch.Tensor) -> None:
        self.first = first
        self.second = second


class OpaqueHolderOutput(nn.Module):
    """Returns a plain object holding two tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> _Holder:
        """Hold two derived tensors."""

        y = self.lin(x)
        return _Holder(y, y.relu())


# Kept by shrink-r4 (per-id subset rule): sole coverage of _fast_live_plans.py, _fast_run.py, _ir/distribution_container.py ....
def test_fast_run_refuses_a_lossy_distribution_output() -> None:
    """A distribution capture flags lossy keeps the honest capability refusal."""

    model = SquashedDistributionOutput()
    x = torch.arange(6.0).view(2, 3)
    trace = tl.trace(model, x)
    with pytest.raises(RunCapabilityUnavailableError) as info:
        trace.run(inputs=x, fast=True)
    assert info.value.fields["code"] == "run_capability_unavailable"
    assert "TransformedDistribution" in str(info.value)
    assert "distribution_state_unrecordable" in str(info.value)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _fast_live_plans.py, _fast_run.py, backends/torch/_outputs.py.
def test_fast_run_names_an_opaque_output_container_as_a_capability_refusal() -> None:
    """A collapsed opaque boundary is refused as unpairable, not as a changed structure."""

    model = OpaqueHolderOutput()
    # Capture's own advisory for an unsupported output container (the BFS
    # fallback) is the documented behaviour here, not the subject.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace = tl.trace(model, torch.arange(6.0).view(2, 3))
        with pytest.raises(RunCapabilityUnavailableError) as info:
            trace.run(inputs=torch.arange(6.0).view(2, 3), fast=True)
    assert info.value.fields["code"] == "run_capability_unavailable"
    assert "_Holder" in str(info.value)


class ClonesItsActivation(nn.Module):
    """Clones an activation, so ``clone`` is a planned functional site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply, clone, add."""

        y = x * 2
        z = y.clone()
        return z + 1
