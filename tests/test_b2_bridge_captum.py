"""Captum bridge: one test per method-class shape, against real Captum (walk U20.4).

``tl.bridge.captum.attribute`` passes ``target=`` only when the caller gives
one (the activation and neuron classes reject it) and, when ``inputs`` is
omitted, forwards every logged input, not only the first. Each case compares
the bridge with the same Captum class called directly on all inputs.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError

captum_attr = pytest.importorskip("captum.attr")

pytestmark = [pytest.mark.optional]


class _TwoInputs(nn.Module):
    """Two positional tensor inputs; ``out_dim=None`` returns one score per row."""

    def __init__(self, out_dim: int | None = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 4)
        self.head = nn.Linear(4, out_dim or 1)
        self.out_dim = out_dim

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenate both inputs and score them."""

        out = self.head(torch.tanh(self.fc(torch.cat([a, b], dim=1))))
        return out if self.out_dim else out.squeeze(1)


def _setup(out_dim: int | None = 3) -> tuple[nn.Module, tuple[torch.Tensor, ...], Any]:
    torch.manual_seed(0)
    model = _TwoInputs(out_dim).eval()
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    log = tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all"))
    return model, inputs, log


def _assert_same(bridged: Any, direct: Any) -> None:
    if isinstance(direct, tuple):
        assert isinstance(bridged, tuple) and len(bridged) == len(direct) == 2
        for got, want in zip(bridged, direct, strict=True):
            assert torch.equal(got, want)
    else:
        assert torch.equal(bridged, direct)


@pytest.mark.parametrize("with_target", [True, False], ids=["target", "no-target"])
def test_primary_attribution_forwards_every_input(with_target: bool) -> None:
    """IntegratedGradients: both inputs reach Captum; target only when given."""

    model, inputs, log = _setup(3 if with_target else None)
    extra = {"target": 1} if with_target else {}
    direct = captum_attr.IntegratedGradients(model).attribute(inputs, n_steps=8, **extra)
    bridged = tl.bridge.captum.attribute(
        log, captum_attr.IntegratedGradients(model), extra.get("target"), n_steps=8
    )
    _assert_same(bridged, direct)


@pytest.mark.parametrize("with_target", [True, False], ids=["target", "no-target"])
def test_layer_attribution_with_and_without_target(with_target: bool) -> None:
    """LayerGradientXActivation on a resolved layer, with and without a target."""

    model, inputs, log = _setup(3 if with_target else None)
    extra = {"target": 2} if with_target else {}
    direct = captum_attr.LayerGradientXActivation(model, model.fc).attribute(inputs, **extra)
    method = captum_attr.LayerGradientXActivation(model, tl.bridge.captum.layer(log, "fc"))
    bridged = tl.bridge.captum.attribute(log, method, **extra)
    _assert_same(bridged, direct)


def test_activation_class_takes_no_target() -> None:
    """LayerActivation has no target parameter; the bridge must not send one."""

    model, inputs, log = _setup()
    direct = captum_attr.LayerActivation(model, model.fc).attribute(inputs)
    bridged = tl.bridge.captum.attribute(log, captum_attr.LayerActivation(model, model.fc))
    _assert_same(bridged, direct)


def test_neuron_class_takes_a_selector_and_no_target() -> None:
    """NeuronGradient takes neuron_selector and no target."""

    model, inputs, log = _setup()
    direct = captum_attr.NeuronGradient(model, model.fc).attribute(inputs, neuron_selector=0)
    bridged = tl.bridge.captum.attribute(
        log, captum_attr.NeuronGradient(model, model.fc), neuron_selector=0
    )
    _assert_same(bridged, direct)


def test_neuron_class_with_a_target() -> None:
    """NeuronConductance takes both neuron_selector and target."""

    model, inputs, log = _setup()
    direct = captum_attr.NeuronConductance(model, model.fc).attribute(
        inputs, neuron_selector=1, target=0, n_steps=8
    )
    bridged = tl.bridge.captum.attribute(
        log, captum_attr.NeuronConductance(model, model.fc), 0, neuron_selector=1, n_steps=8
    )
    _assert_same(bridged, direct)


@pytest.mark.parametrize("with_target", [True, False], ids=["target", "no-target"])
def test_wrapper_class_forwards_target_through_kwargs(with_target: bool) -> None:
    """NoiseTunnel(Saliency) receives target only when given."""

    model, inputs, log = _setup(3 if with_target else None)
    extra = {"target": 1} if with_target else {}
    options = {"nt_type": "smoothgrad", "nt_samples": 3, "stdevs": 0.1, **extra}
    torch.manual_seed(7)
    direct = captum_attr.NoiseTunnel(captum_attr.Saliency(model)).attribute(inputs, **options)
    torch.manual_seed(7)
    bridged = tl.bridge.captum.attribute(
        log, captum_attr.NoiseTunnel(captum_attr.Saliency(model)), **options
    )
    _assert_same(bridged, direct)


def test_single_input_forwards_a_bare_tensor() -> None:
    """A one-input forward hands Captum the tensor itself, as a direct call would."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 2)).eval()
    x = torch.randn(2, 3)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    direct = captum_attr.Saliency(model).attribute(x, target=0)
    bridged = tl.bridge.captum.attribute(log, captum_attr.Saliency(model), 0)
    _assert_same(bridged, direct)


class _KeywordInput(nn.Module):
    """A forward whose second tensor is keyword-only."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, a: torch.Tensor, *, b: torch.Tensor) -> torch.Tensor:
        """Add the keyword-only tensor."""

        return self.fc(a + b)


def test_keyword_only_input_refuses_instead_of_dropping_it() -> None:
    """An input Captum cannot pass positionally is refused, never silently dropped."""

    model = _KeywordInput().eval()
    log = tl.trace(
        model,
        (torch.randn(2, 3),),
        {"b": torch.randn(2, 3)},
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.captum.attribute(log, captum_attr.Saliency(model), 0)
    assert info.value.fields["code"] == "bridge_captum_inputs_unresolved"
