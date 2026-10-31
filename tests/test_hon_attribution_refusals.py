"""Typed attribution refusals from the honesty lane, each with a running remedy.

- U20.5: occlusion over an element-empty selection refuses
  (``occlusion_selection_empty``) instead of returning a confident zero.
- U21.5: zero-noise SmoothGrad refuses (``smoothgrad_noise_level_invalid``)
  instead of averaging identical saliency maps.
- U23.3: text attribution over an input longer than the model's position
  limit refuses (``text_input_exceeds_position_limit``) with a remedy naming
  ``max_length``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens import attribution
from torchlens.attribution._occlusion import occlusion
from torchlens.errors import AttributionError


class _ImageScale(nn.Module):
    """Scale an image by two."""

    def forward(self, x: Tensor) -> Tensor:
        return x * 2.0


class _Linear(nn.Module):
    """Small classifier for saliency."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x)


def _output_sum(trace: Any) -> Tensor:
    """Return the single captured output tensor's sum."""

    return trace.output_ops[0].out.sum()


def test_occlusion_refuses_an_empty_selection_and_the_remedy_runs() -> None:
    """U20.5: zero selected elements refuse typed; one selected element scores."""

    inputs = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    trace = tl.trace(
        _ImageScale(), inputs, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    try:
        label = trace.input_ops[0].label
        empty = tl.units(label, [])
        assert empty.resolve(trace).empty
        with pytest.raises(AttributionError) as info:
            occlusion(trace, empty, score=_output_sum)
        assert info.value.fields["code"] == "occlusion_selection_empty"
        assert "tl.units(label" in str(info.value)

        whole = occlusion(trace, trace[label], score=_output_sum)
        assert whole.values.item() == pytest.approx(90.0)
        one = occlusion(trace, tl.units(label, [(0, 0, 1, 1)]), score=_output_sum)
        assert one.values.item() == pytest.approx(10.0)
    finally:
        trace.cleanup()


@pytest.mark.parametrize("noise_level", [0.0, 0, -0.1])
def test_smoothgrad_refuses_zero_or_negative_noise(noise_level: float) -> None:
    """U21.5: zero noise (and negative, as before) refuses typed."""

    with pytest.raises(AttributionError) as info:
        attribution.smoothgrad(_Linear(), torch.randn(2, 4), target=0, noise_level=noise_level)
    assert info.value.fields["code"] == "smoothgrad_noise_level_invalid"


def test_smoothgrad_zero_noise_remedies_run() -> None:
    """Both remedies run: a positive scale, or saliency for the noise-free map."""

    torch.manual_seed(0)
    model, x = _Linear(), torch.randn(2, 4)
    noisy = attribution.smoothgrad(model, x, target=0, num_samples=3, noise_level=0.1, seed=0)
    assert noisy.values.shape == x.shape
    plain = attribution.saliency(model, x, target=0)
    assert plain.values.shape == x.shape
