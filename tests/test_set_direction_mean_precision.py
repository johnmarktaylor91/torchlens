"""``set_direction_mean`` keeps the site's precision.

A float64 site value gets a float64 unit direction and mean coefficient, so the
result and its gradient match a float64 hand computation. Every other dtype keeps
the float32 unit and coefficient it always used, bit for bit.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import reference, set_direction_mean

_DIRECTION = torch.tensor([1.0, 2.0, -1.0, 0.5, 0.25], dtype=torch.float64) / 3.0


class _Net(nn.Module):
    """``lin2(lin1(x))``: the edit sits on ``lin1``'s output."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(5, 5).to(dtype)
        self.lin2 = nn.Linear(5, 2).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 5, generator=generator, dtype=torch.float64).to(dtype)
    donors = torch.randn(4, 3, 5, generator=generator, dtype=torch.float64).to(dtype)
    return x, donors


def _edited_and_grad(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, nn.Module]:
    model = _Net(dtype).eval()
    x, donors = _inputs(dtype)
    edit = set_direction_mean(
        _DIRECTION.to(dtype), reference(donors, origin="donors"), feature_axis=-1
    )
    trace = tl.trace(model, x, intervene=tl.when(tl.module("lin1"), edit))
    output = trace[trace.output_layers[0]].out
    output.sum().backward()
    assert model.lin1.weight.grad is not None
    return trace["linear_1_1"].out.detach(), model.lin1.weight.grad.clone(), model


def test_float64_site_matches_float64_hand_value_and_gradient() -> None:
    edited, grad, _ = _edited_and_grad(torch.float64)
    model = _Net(torch.float64).eval()
    x, donors = _inputs(torch.float64)
    unit = _DIRECTION / _DIRECTION.norm()
    mean_coef = (donors * unit).sum(dim=-1).mean()
    site = model.lin1(x)
    hand = site - (site @ unit)[..., None] * unit + mean_coef.detach() * unit
    assert torch.allclose(edited, hand.detach(), rtol=0.0, atol=1e-12)
    model.lin2(hand).sum().backward()
    assert model.lin1.weight.grad is not None
    assert torch.allclose(grad, model.lin1.weight.grad, rtol=1e-10, atol=1e-12)


def test_float32_site_is_unchanged_bit_for_bit() -> None:
    """The float32 path keeps its float32 unit and coefficient exactly."""

    edited, _, _ = _edited_and_grad(torch.float32)
    model = _Net(torch.float32).eval()
    x, donors = _inputs(torch.float32)
    host = _DIRECTION.to(torch.float32)
    unit = host / host.norm()
    mean_coef = float(torch.stack([(member * unit).sum(dim=-1).mean() for member in donors]).mean())
    with torch.no_grad():
        site = model.lin1(x)
        projection = (site * unit).sum(dim=-1, keepdim=True) * unit
        hand = site - projection + mean_coef * unit
    assert torch.equal(edited, hand)


def test_recorded_coefficient_is_the_one_applied() -> None:
    """The edit's kwargs keep the float32 coefficient; a float64 site applies float64."""

    x, donors = _inputs(torch.float64)
    edit = set_direction_mean(_DIRECTION, reference(donors, origin="donors"), feature_axis=-1)
    coefficient = dict(edit.kwargs)["mean_coefficient"]
    unit32 = _DIRECTION.float() / _DIRECTION.float().norm()
    per_member = [(member.float() * unit32).sum(dim=-1).mean() for member in donors]
    assert coefficient == float(torch.stack(per_member).mean())
    model = copy.deepcopy(_Net(torch.float64)).eval()
    trace = tl.trace(model, x, intervene=tl.when(tl.module("lin1"), edit))
    unit = _DIRECTION / _DIRECTION.norm()
    along = (trace["linear_1_1"].out @ unit).detach()
    expected = float((donors * unit).sum(dim=-1).mean())
    assert torch.allclose(along, torch.full_like(along, expected), rtol=0.0, atol=1e-12)
