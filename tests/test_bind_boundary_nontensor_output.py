"""A bound module-boundary rule leaves a module output it cannot address whole.

Found by the behaviour net's bind cell on a policy whose head returns a
distribution object: the bound executor's output walker fell back to a
breadth-first search, found the object's tensors with no stable path, and
``bind`` handed the bare tensor back in the object's place, so the model crashed
on ``.stddev``. The capture door never saw a tensor site in such an output and
recorded zero fire. Both doors now agree: a module output that is a tensor or a
supported container is edited at its tensor leaves; any other object holds no
addressable site and the rule records zero fire there.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import BindingRuntimeError, SiteResolutionError


class _Gaussian:
    """A distribution-like object: two tensors behind attributes, not a container."""

    def __init__(self, mean: torch.Tensor, stddev: torch.Tensor) -> None:
        self.mean = mean
        self.stddev = stddev


class _DistHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Linear(3, 2)
        self.log_std = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> _Gaussian:
        return _Gaussian(self.mean(x), torch.exp(self.log_std(x)))


class _TupleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Linear(3, 2)
        self.log_std = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(x), torch.exp(self.log_std(x))


class _Policy(nn.Module):
    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.body = nn.Linear(4, 3)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.head(torch.tanh(self.body(x)))
        mean, stddev = (dist.mean, dist.stddev) if isinstance(dist, _Gaussian) else dist
        return mean + 0.5 * stddev


def _build(head: type[nn.Module]) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = _Policy(head()).eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        plain = model(x)
    return model, x, plain


@pytest.mark.smoke
def test_bound_boundary_rule_leaves_an_unaddressable_output_whole() -> None:
    model, x, plain = _build(_DistHead)
    bound = tl.when(tl.module("head"), tl.scale(0.5)).bind(model, on_zero_fire="disclose")
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        out = bound(x)
    assert torch.equal(out, plain)
    assert bound.last_report.zero_fire_rule_ids


def test_bound_boundary_rule_on_an_unaddressable_output_refuses_zero_fire_typed() -> None:
    model, x, _plain = _build(_DistHead)
    bound = tl.when(tl.module("head"), tl.scale(0.5)).bind(model, on_zero_fire="error")
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        with pytest.raises(BindingRuntimeError) as info:
            bound(x)
    assert info.value.fields["code"] == "bind_zero_fire"


def test_capture_door_records_zero_fire_on_the_same_output() -> None:
    model, x, _plain = _build(_DistHead)
    spec = tl.when(tl.module("head"), tl.scale(0.5))
    with warnings.catch_warnings():
        # The capture's own output walker warns once that it cannot address the
        # object (the pre-existing "falling back to BFS" advisory); the test is
        # about the door's zero-fire verdict, not that advisory.
        warnings.simplefilter("ignore")
        with pytest.raises(SiteResolutionError) as info:
            tl.trace(model, x, intervene=spec, on_zero_fire="error")
    assert info.value.fields["code"] == "trace_zero_fire"


@pytest.mark.smoke
def test_bound_boundary_rule_still_edits_supported_container_leaves() -> None:
    model, x, plain = _build(_TupleHead)
    spec = tl.when(tl.module("head"), tl.scale(0.5))
    bound = spec.bind(model)
    with torch.no_grad():
        out = bound(x)
    assert not torch.equal(out, plain)
    assert not bound.last_report.zero_fire_rule_ids
    with torch.no_grad():
        steered = tl.trace(model, x, intervene=spec, on_zero_fire="error")
    assert torch.equal(out, steered.reconstruct_output())
