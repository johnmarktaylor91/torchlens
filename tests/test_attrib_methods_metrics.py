"""F06 B7: infidelity + sensitivity metrics (attrib memo D14-D15).

Analytic infidelity rows (a linear model's gradient attribution is exactly
faithful, so infidelity is zero), the named perturbation policies and the
callable escape, the out-of-range disclosure warning, the layer/CAM shape
refusal with the expansion recipe, and the three-rung sensitivity determinism
ladder including the behavioral probe.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution._result import AttributionWarning
from torchlens.errors import AttributionError


class _Linear(nn.Module):
    """Linear model: gradient attributions are exactly infidelity-faithful."""

    def __init__(self) -> None:
        """Build fixed weights."""

        super().__init__()
        self.head = nn.Linear(4, 2, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor(
                    [[0.5, -1.0, 2.0, 0.25], [1.5, 0.25, -0.75, -0.5]], dtype=torch.float64
                )
            )
            self.head.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))

    def forward(self, x: Tensor) -> Tensor:
        """Linear forward."""

        return self.head(x)


class _SpatialNet(nn.Module):
    """Tiny conv net for square-removal and CAM-mismatch rows."""

    def __init__(self) -> None:
        """Build a fixed conv + head."""

        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1, dtype=torch.float64)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(2 * 3 * 9, dtype=torch.float64).reshape(2, 3, 3, 3) / 50.0
            )
            self.conv.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Conv, spatial mean, two logits."""

        return self.conv(x).mean(dim=(2, 3))


def test_infidelity_square_removal_and_refusals() -> None:
    """square_removal works on spatial inputs and refuses on flat ones."""

    model = _SpatialNet()
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    grad = torch.autograd.grad(model(x.requires_grad_(True))[..., 0].sum(), x)[0]
    result = attribution.infidelity(
        model, x.detach(), target=0, attribution=grad, perturb="square_removal", seed=4
    )
    assert result.value == pytest.approx(0.0, abs=1e-18)  # conv+mean is linear
    assert "square_removal" in result.settings["perturb"]

    with pytest.raises(AttributionError) as excinfo:
        attribution.infidelity(
            _Linear(),
            torch.zeros(1, 4, dtype=torch.float64),
            target=0,
            attribution=torch.zeros(1, 4, dtype=torch.float64),
            perturb="square_removal",
        )
    assert excinfo.value.fields["code"] == "metric_perturbation_invalid"
    with pytest.raises(AttributionError) as unknown:
        attribution.infidelity(
            _Linear(),
            torch.zeros(1, 4, dtype=torch.float64),
            target=0,
            attribution=torch.zeros(1, 4, dtype=torch.float64),
            perturb="banana",
        )
    assert unknown.value.fields["code"] == "metric_perturbation_invalid"


def test_infidelity_callable_escape_contract() -> None:
    """A compatible callable works; contract violations refuse typed."""

    model = _Linear()
    x = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
    exact = model.head.weight[0].reshape(1, 4)

    def fixed_perturb(leaves: tuple[Tensor, ...]) -> tuple[tuple, tuple]:
        """Deterministic fixed perturbation of the one leaf."""

        dx = torch.full_like(leaves[0], 0.01)
        return (dx,), (leaves[0] - dx,)

    result = attribution.infidelity(
        model, x, target=0, attribution=exact, perturb=fixed_perturb, num_samples=2
    )
    assert result.value == pytest.approx(0.0, abs=1e-20)
    assert "callable" in result.settings["perturb"]

    def broken_perturb(leaves: tuple[Tensor, ...]) -> Tensor:
        """Violates the (perturbations, perturbed) contract."""

        return leaves[0]

    with pytest.raises(AttributionError) as excinfo:
        attribution.infidelity(model, x, target=0, attribution=exact, perturb=broken_perturb)
    assert excinfo.value.fields["code"] == "metric_perturbation_invalid"


def test_infidelity_out_of_range_warns_coded() -> None:
    """D14: perturbations leaving the observed input range WARN, never refuse."""

    model = _Linear()
    x = torch.full((1, 4), 0.5, dtype=torch.float64)  # zero-width value hull
    exact = model.head.weight[0].reshape(1, 4)
    with pytest.warns(AttributionWarning) as captured:
        result = attribution.infidelity(
            model, x, target=0, attribution=exact, noise_std=0.1, seed=5
        )
    assert result.value == pytest.approx(0.0, abs=1e-18)
    codes = {warning.message.fields.get("code") for warning in captured}
    assert "metric_perturbation_out_of_range" in codes


def test_infidelity_layer_and_cam_shape_mismatch_refuse_with_recipe() -> None:
    """Layer-space values and native CAMs refuse with the expansion recipe."""

    model = _SpatialNet()
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    cam_shaped = torch.rand(1, 1, 8, 8, dtype=torch.float64)  # channel mismatch
    with pytest.raises(AttributionError) as excinfo:
        attribution.infidelity(model, x, target=0, attribution=cam_shaped)
    assert excinfo.value.fields["code"] == "metric_attribution_shape_mismatch"
    assert "input space" in str(excinfo.value)


def test_sensitivity_rung1_refuses_unseeded_stochastic_before_work() -> None:
    """A known-stochastic method without a seed refuses BEFORE any forward."""

    class _NeverCalled(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            raise AssertionError("rung 1 must refuse before any forward runs")

    with pytest.raises(AttributionError) as excinfo:
        attribution.sensitivity(
            torch.zeros(1, 4, dtype=torch.float64),
            method=attribution.smoothgrad,
            model=_NeverCalled(),
            target=0,
        )
    assert excinfo.value.fields["code"] == "metric_determinism_unfrozen"


def test_sensitivity_seeded_stochastic_passes_structurally() -> None:
    """Rung 1 composes: a seeded stochastic method runs without a probe."""

    model = _Linear()
    x = torch.tensor([[0.2, -0.1, 0.4, 0.3]], dtype=torch.float64)
    result = attribution.sensitivity(
        x,
        method=attribution.smoothgrad,
        model=model,
        target=0,
        method_kwargs={"seed": 11, "num_samples": 3},
        num_samples=3,
        seed=6,
    )
    assert result.metric == "sensitivity_max"
    assert result.extra["determinism_probe"] == "structural"
    # A linear model's |gradient| is input-independent: sensitivity 0.
    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_sensitivity_rung2_refuses_disclosed_unseeded_provenance() -> None:
    """An opaque callable disclosing unseeded stochastic provenance refuses."""

    model = _Linear()

    def opaque(inputs: Tensor, input_kwargs: dict | None) -> attribution.Attribution:
        """Unseeded smoothgrad hidden behind a closed callable."""

        return attribution.smoothgrad(model, inputs, input_kwargs, target=0)

    with pytest.raises(AttributionError) as excinfo:
        attribution.sensitivity(
            torch.zeros(1, 4, dtype=torch.float64), attribute=opaque, num_samples=2
        )
    assert excinfo.value.fields["code"] == "metric_determinism_unfrozen"


def test_sensitivity_rung3_behavioral_probe() -> None:
    """A hidden-noise opaque callable fails the probe; a clean one passes."""

    model = _Linear()

    def noisy(inputs: Tensor, input_kwargs: dict | None) -> attribution.Attribution:
        """Opaque callable with UNDISCLOSED randomness (the probe's quarry)."""

        del input_kwargs
        values = model.head.weight[0].reshape(1, 4) + 0.01 * torch.randn(1, 4, dtype=torch.float64)
        return attribution.Attribution(
            method="mystery", values=values, target_repr="index=0", extra={}
        )

    with pytest.raises(AttributionError) as excinfo:
        attribution.sensitivity(
            torch.zeros(1, 4, dtype=torch.float64), attribute=noisy, num_samples=2
        )
    assert excinfo.value.fields["code"] == "metric_determinism_probe_failed"
    assert excinfo.value.fields["observed_relative_difference"] > 1e-6

    def clean(inputs: Tensor, input_kwargs: dict | None) -> attribution.Attribution:
        """Deterministic opaque callable."""

        return attribution.saliency(model, inputs, input_kwargs, target=0)

    result = attribution.sensitivity(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64),
        attribute=clean,
        num_samples=2,
        seed=7,
    )
    assert result.extra["determinism_probe"] == "passed_not_proven"
    assert "observed_two_call_relative_difference" in result.extra
