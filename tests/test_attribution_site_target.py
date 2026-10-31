"""Site targets: attribution to a hidden layer's output or one unit of it.

Every value is checked against an independent computation: autograd of the
selected hidden activation, written out by hand without hooks.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens.attribution import AttributionError, SiteTarget

_ATOL = 1e-6
_UNIT = 4


class _MLP(nn.Module):
    """Two linear layers with a tanh between them."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 7)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(7, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Score ``x``."""

        return self.fc2(self.act(self.fc1(x)))


class _MLPWithConstantBranch(_MLP):
    """An MLP plus a branch computed from a buffer only, never from the input."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Linear(2, 3)
        self.register_buffer("anchor", torch.ones(1, 2))

    def forward(self, x: Tensor) -> Tensor:
        """Score ``x`` and add the input-free offset."""

        return super().forward(x) + self.offset(self.anchor)


class _ReusedActivation(_MLP):
    """An MLP that applies its tanh module twice."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``act`` before and after ``fc1``."""

        return self.fc2(self.act(self.fc1(self.act(x))))


def _mlp_and_input() -> tuple[_MLP, Tensor]:
    """Return a seeded MLP in eval mode and a batch of two inputs."""

    torch.manual_seed(0)
    return _MLP().eval(), torch.randn(2, 5)


def _hand_gradient(model: _MLP, x: Tensor, select: str) -> Tensor:
    """Return d(selected hidden activation, summed) / dx by plain autograd."""

    leaf = x.detach().clone().requires_grad_(True)
    hidden = torch.tanh(model.fc1(leaf))
    scalar = hidden[..., _UNIT].sum() if select == "unit" else hidden.sum()
    (gradient,) = torch.autograd.grad(scalar, leaf)
    return gradient


def _hand_integrated_gradients(model: _MLP, x: Tensor, n_steps: int) -> Tensor:
    """Midpoint-rule Integrated Gradients of the hidden unit from a zero baseline."""

    gradients = [
        _hand_gradient(model, ((step + 0.5) / n_steps) * x, "unit") for step in range(n_steps)
    ]
    return x * torch.stack(gradients).mean(dim=0)


@pytest.mark.parametrize("method_name", ["input_x_grad", "integrated_gradients", "saliency"])
def test_site_target_unit_matches_hand_autograd(method_name: str) -> None:
    """Attribution to one hidden unit equals the hand computation."""

    model, x = _mlp_and_input()
    target = SiteTarget("act", unit=_UNIT)
    if method_name == "integrated_gradients":
        result = tl.attribution.integrated_gradients(model, x, target=target, num_steps=16)
        expected = _hand_integrated_gradients(model, x, 16)
    else:
        result = getattr(tl.attribution, method_name)(model, x, target=target)
        gradient = _hand_gradient(model, x, "unit")
        expected = gradient.abs() if method_name == "saliency" else gradient * x
    torch.testing.assert_close(result.values, expected, atol=_ATOL, rtol=0)
    assert bool(expected.abs().sum() > 0)
    assert result.target_repr == "SiteTarget('act', unit=4)"


def test_site_target_tuple_unit_and_whole_site_match_hand_autograd() -> None:
    """A tuple unit indexes after the batch axis; no unit attributes the whole output."""

    model, x = _mlp_and_input()
    by_tuple = tl.attribution.saliency(model, x, target=SiteTarget("act", unit=(_UNIT,)))
    torch.testing.assert_close(
        by_tuple.values, _hand_gradient(model, x, "unit").abs(), atol=_ATOL, rtol=0
    )
    whole = tl.attribution.saliency(model, x, target=SiteTarget("act"))
    torch.testing.assert_close(
        whole.values, _hand_gradient(model, x, "whole").abs(), atol=_ATOL, rtol=0
    )


def test_site_target_layer_method_matches_hand_autograd() -> None:
    """layer_attribution of fc1 toward a downstream unit equals activation times gradient."""

    model, x = _mlp_and_input()
    result = tl.attribution.layer_attribution(
        model, x, target=SiteTarget("act", unit=_UNIT), layer="fc1"
    )
    activation = model.fc1(x).detach().requires_grad_(True)
    (gradient,) = torch.autograd.grad(torch.tanh(activation)[..., _UNIT].sum(), activation)
    torch.testing.assert_close(result.values, activation.detach() * gradient, atol=_ATOL, rtol=0)


def test_site_target_on_integer_ids_matches_float_embedding_run() -> None:
    """Both seams together: a site target from token ids equals the float run."""

    from tests.test_attribution_integer_inputs import _token_pair

    token_model, float_model, ids = _token_pair()
    target = SiteTarget("body.encoder", unit=2)
    from_ids = tl.attribution.layer_integrated_gradients(
        token_model, ids, target=target, layer="body.encoder.norm1", num_steps=8
    )
    from_floats = tl.attribution.layer_integrated_gradients(
        float_model,
        token_model.embed(ids).detach(),
        target=target,
        layer="body.encoder.norm1",
        num_steps=8,
    )
    torch.testing.assert_close(from_ids.values, from_floats.values, atol=_ATOL, rtol=0)
    assert bool(from_ids.values.abs().sum() > 0)


def _target_remedy(error: AttributionError) -> SiteTarget:
    """Evaluate the ``target=`` spelling a refusal gives."""

    match = re.search(r"target=(tl\.attribution\.SiteTarget\(.*?\))[.,]? ", str(error))
    assert match is not None, str(error)
    spelled = eval(match.group(1), {"tl": tl})  # noqa: S307
    assert isinstance(spelled, SiteTarget)
    return spelled


def test_site_target_refuses_missing_site_and_its_remedy_runs() -> None:
    """A site name the model lacks refuses with the closest real name."""

    model, x = _mlp_and_input()
    with pytest.raises(AttributionError) as caught:
        tl.attribution.saliency(model, x, target=SiteTarget("fc9", unit=1))
    assert caught.value.fields.get("code") == "attribution_site_not_found"
    assert "'fc9'" in str(caught.value)
    remedy = _target_remedy(caught.value)
    assert remedy.module_address in {"fc1", "fc2"}
    assert tuple(tl.attribution.saliency(model, x, target=remedy).values.shape) == (2, 5)


def test_site_target_refuses_site_independent_of_input_and_its_remedy_runs() -> None:
    """A site computed without the input refuses rather than returning zeros."""

    torch.manual_seed(0)
    model = _MLPWithConstantBranch().eval()
    x = torch.randn(2, 5)
    with pytest.raises(AttributionError) as caught:
        tl.attribution.saliency(model, x, target=SiteTarget("offset"))
    assert caught.value.fields.get("code") == "attribution_site_independent_of_input"
    remedy = _target_remedy(caught.value)
    assert remedy == SiteTarget("fc2")
    values = tl.attribution.saliency(model, x, target=remedy).values
    assert bool(values.abs().sum() > 0)


def test_site_target_refuses_bad_unit_and_reused_site_and_remedies_run() -> None:
    """An out-of-range unit and an unqualified reused site refuse; their remedies run."""

    model, x = _mlp_and_input()
    with pytest.raises(AttributionError) as caught:
        tl.attribution.saliency(model, x, target=SiteTarget("act", unit=99))
    assert caught.value.fields.get("code") == "attribution_site_unit_invalid"
    remedy = _target_remedy(caught.value)
    assert remedy == SiteTarget("act", unit=0)
    tl.attribution.saliency(model, x, target=remedy)

    torch.manual_seed(0)
    reused = _ReusedActivation().eval()
    x7 = torch.randn(2, 5)
    with pytest.raises(AttributionError) as caught:
        tl.attribution.saliency(reused, x7, target=SiteTarget("act", unit=1))
    assert caught.value.fields.get("code") == "attribution_site_firing_invalid"
    remedy = _target_remedy(caught.value)
    assert remedy == SiteTarget("act", unit=1, call=0)
    first_call = tl.attribution.saliency(reused, x7, target=remedy).values
    leaf = x7.detach().clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(torch.tanh(leaf)[..., 1].sum(), leaf)
    torch.testing.assert_close(first_call, gradient.abs(), atol=_ATOL, rtol=0)

    with pytest.raises(AttributionError) as caught:
        SiteTarget("act", unit=1.5)  # type: ignore[arg-type]
    assert caught.value.fields.get("code") == "attribution_site_unit_invalid"


def test_site_target_refuses_where_unsupported() -> None:
    """A method that scores only the model output refuses a site target by name."""

    model, x = _mlp_and_input()
    target = SiteTarget("act", unit=1)
    attribution = tl.attribution.saliency(model, x, target=1)
    with pytest.raises(AttributionError) as caught:
        tl.attribution.infidelity(model, x, target=target, attribution=attribution)
    assert caught.value.fields.get("code") == "attribution_site_target_unsupported"
    assert str(caught.value).startswith("infidelity cannot attribute a SiteTarget")
    spelling = re.search(r"e\.g\. (tl\.attribution\.saliency\(.*?\))\. See", str(caught.value))
    assert spelling is not None
    eval(spelling.group(1), {"tl": tl, "model": model, "inputs": x})  # noqa: S307
