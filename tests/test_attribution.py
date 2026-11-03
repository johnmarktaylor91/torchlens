"""Tests for native TorchLens input attribution."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens
import torchlens.attribution as attribution


class TinyLinear(nn.Module):
    """Single linear layer with fixed weights for closed-form gradient checks."""

    def __init__(self, weight: Tensor) -> None:
        """Initialize the model with a known weight matrix.

        Parameters
        ----------
        weight
            Weight matrix with shape ``(out_features, in_features)``.
        """

        super().__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the linear layer.

        Parameters
        ----------
        inputs
            Input tensor with shape ``(..., in_features)``.

        Returns
        -------
        Tensor
            Linear outputs.
        """

        return self.linear(inputs)


class SmoothTinyMlp(nn.Module):
    """Small smooth MLP used for Integrated Gradients completeness."""

    def __init__(self) -> None:
        """Initialize deterministic double-precision layers."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 4),
            nn.Softplus(beta=1.5),
            nn.Linear(4, 2),
        )
        self.double()
        with torch.no_grad():
            first = self.net[0]
            second = self.net[2]
            assert isinstance(first, nn.Linear)
            assert isinstance(second, nn.Linear)
            first.weight.copy_(
                torch.tensor(
                    [
                        [0.20, -0.40, 0.10],
                        [0.35, 0.15, -0.25],
                        [-0.30, 0.45, 0.20],
                        [0.10, -0.20, 0.30],
                    ],
                    dtype=torch.float64,
                )
            )
            first.bias.copy_(torch.tensor([0.05, -0.10, 0.20, 0.15], dtype=torch.float64))
            second.weight.copy_(
                torch.tensor(
                    [
                        [0.40, -0.35, 0.25, 0.10],
                        [-0.20, 0.30, 0.15, -0.45],
                    ],
                    dtype=torch.float64,
                )
            )
            second.bias.copy_(torch.tensor([0.10, -0.05], dtype=torch.float64))

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the MLP.

        Parameters
        ----------
        inputs
            Input tensor with shape ``(..., 3)``.

        Returns
        -------
        Tensor
            Output tensor with shape ``(..., 2)``.
        """

        return self.net(inputs)


class SquareTarget(nn.Module):
    """Nonlinear scalar-output model for SmoothGrad seed checks."""

    def forward(self, inputs: Tensor) -> Tensor:
        """Return per-sample squared sums.

        Parameters
        ----------
        inputs
            Input tensor.

        Returns
        -------
        Tensor
            Per-sample scalar outputs.
        """

        return inputs.square().sum(dim=-1, keepdim=True)


class TwoInputLinear(nn.Module):
    """Two-input linear model for multi-leaf attribution tests."""

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        """Combine two input tensors into two logits.

        Parameters
        ----------
        first
            First input tensor with shape ``N, 2``.
        second
            Second input tensor with shape ``N, 2``.

        Returns
        -------
        Tensor
            Output tensor with shape ``N, 2``.
        """

        first_logit = (2.0 * first[..., 0] - 3.0 * second[..., 0]).unsqueeze(-1)
        second_logit = (first[..., 1] + 4.0 * second[..., 1]).unsqueeze(-1)
        return torch.cat((first_logit, second_logit), dim=-1)


class TwoInputSmooth(nn.Module):
    """Smooth two-input scalar model for Integrated Gradients completeness."""

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        """Return a smooth scalar-like output from two tensors.

        Parameters
        ----------
        first
            First input tensor.
        second
            Second input tensor.

        Returns
        -------
        Tensor
            Output tensor with shape ``1, 1``.
        """

        value = (
            torch.nn.functional.softplus(first + 0.5 * second).sum()
            + (first * second).sum()
            + 0.25 * second.square().sum()
        )
        return value.reshape(1, 1)


class KwargScaleModel(nn.Module):
    """Model that consumes an attributed tensor keyword argument."""

    def forward(self, inputs: Tensor, *, scale: Tensor, offset: float = 0.0) -> Tensor:
        """Scale inputs with a tensor kwarg and add a non-attributed offset.

        Parameters
        ----------
        inputs
            Positional input tensor.
        scale
            Keyword tensor input.
        offset
            Non-tensor keyword offset.

        Returns
        -------
        Tensor
            Output tensor with shape matching ``inputs``.
        """

        return inputs * scale + offset


def _sum_attribution_values(values: object) -> Tensor:
    """Sum every tensor leaf in an attribution value tree.

    Parameters
    ----------
    values
        Bare tensor or nested attribution values.

    Returns
    -------
    Tensor
        Scalar sum over tensor leaves.
    """

    if isinstance(values, Tensor):
        return values.sum()
    if isinstance(values, tuple | list):
        return sum(_sum_attribution_values(value) for value in values if value is not None)
    if isinstance(values, dict):
        return sum(_sum_attribution_values(value) for value in values.values() if value is not None)
    return torch.tensor(0.0)


def test_attribution_result_repr_summarizes_values() -> None:
    """AttributionResult repr should not inline tensor values."""

    weight = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 3.0, -4.0]])
    model = TinyLinear(weight)
    inputs = torch.randn(4, 3)

    rendered = repr(attribution.saliency(model, inputs, target=1))

    assert "Tensor(shape=(4, 3)" in rendered
    assert "tensor([" not in rendered


def test_input_attribution_supports_tensor_kwargs() -> None:
    """Tensor keyword arguments are attributed and non-tensors pass through."""

    model = KwargScaleModel()
    inputs = torch.tensor([[1.0, -2.0]])
    scale = torch.tensor([[3.0, -4.0]])

    result = attribution.input_x_grad(
        model,
        inputs,
        {"scale": scale, "offset": 1.5},
        target=lambda output: output.sum(),
    )

    assert isinstance(result.values, dict)
    input_values = result.values["inputs"]
    kwarg_values = result.values["input_kwargs"]
    assert isinstance(input_values, tuple)
    assert isinstance(input_values[0], Tensor)
    assert isinstance(kwarg_values, dict)
    assert isinstance(kwarg_values["scale"], Tensor)
    assert kwarg_values["offset"] is None
    torch.testing.assert_close(input_values[0], inputs * scale)
    torch.testing.assert_close(kwarg_values["scale"], inputs * scale)


def test_integrated_gradients_rejects_mismatched_baseline_structure() -> None:
    """Structured baselines must mirror attributed input leaves."""

    model = TwoInputLinear()
    first = torch.tensor([[0.5, -1.0]])
    second = torch.tensor([[2.0, 3.0]])

    with pytest.raises(torchlens.errors.AttributionError, match="baseline must mirror"):
        attribution.integrated_gradients(
            model,
            (first, second),
            target=0,
            baseline=(torch.zeros_like(first),),
        )
