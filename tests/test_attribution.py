"""Tests for native TorchLens input attribution."""

from __future__ import annotations

import torch
import pytest
from torch import Tensor, nn

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


def test_saliency_matches_linear_weight_oracle() -> None:
    """Saliency for ``y = W x`` equals the absolute selected row of ``W``."""

    weight = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 3.0, -4.0]])
    model = TinyLinear(weight)
    inputs = torch.tensor([[0.2, -0.3, 0.5], [1.0, 2.0, -1.0]])

    result = attribution.saliency(model, inputs, target=1)

    expected = weight[1].abs().expand_as(inputs)
    assert result.method == "saliency"
    assert result.values.shape == inputs.shape
    torch.testing.assert_close(result.values, expected)


def test_input_x_grad_shape_sign_and_finiteness() -> None:
    """Input-times-gradient preserves shape and expected signs on a linear model."""

    weight = torch.tensor([[1.0, -2.0, 0.5]])
    model = TinyLinear(weight)
    inputs = torch.tensor([[2.0, 3.0, -4.0]])

    result = attribution.input_x_grad(model, inputs, target=0)

    expected = inputs * weight[0]
    assert result.values.shape == inputs.shape
    assert torch.isfinite(result.values).all()
    torch.testing.assert_close(result.values, expected)
    assert torch.equal(torch.sign(result.values), torch.sign(expected))


def test_integrated_gradients_completeness_on_smooth_mlp() -> None:
    """Integrated Gradients sum matches target difference from baseline to input."""

    model = SmoothTinyMlp()
    inputs = torch.tensor([[0.60, -0.20, 0.40]], dtype=torch.float64)
    baseline = torch.tensor([[-0.10, 0.05, 0.20]], dtype=torch.float64)

    result = attribution.integrated_gradients(
        model,
        inputs,
        target=1,
        baseline=baseline,
        n_steps=256,
    )

    with torch.no_grad():
        target_delta = model(inputs)[..., 1].sum() - model(baseline)[..., 1].sum()
    attribution_sum = result.values.sum()
    residual = (attribution_sum - target_delta).abs()

    assert result.values.shape == inputs.shape
    assert residual <= 1e-4
    torch.testing.assert_close(attribution_sum, target_delta, rtol=1e-3, atol=1e-4)


def test_smoothgrad_seed_determinism() -> None:
    """SmoothGrad is deterministic for a fixed seed and changes for another seed."""

    model = SquareTarget()
    inputs = torch.tensor([[0.7, -1.2, 0.4]])

    first = attribution.smoothgrad(
        model,
        inputs,
        target=lambda output: output.sum(),
        n_samples=12,
        noise_level=0.35,
        seed=123,
    )
    second = attribution.smoothgrad(
        model,
        inputs,
        target=lambda output: output.sum(),
        n_samples=12,
        noise_level=0.35,
        seed=123,
    )
    different = attribution.smoothgrad(
        model,
        inputs,
        target=lambda output: output.sum(),
        n_samples=12,
        noise_level=0.35,
        seed=456,
    )

    torch.testing.assert_close(first.values, second.values)
    assert not torch.allclose(first.values, different.values)


def test_saliency_supports_two_positional_tensor_inputs() -> None:
    """Saliency returns one attribution tensor per positional tensor input."""

    model = TwoInputLinear()
    first = torch.tensor([[0.5, -1.0]])
    second = torch.tensor([[2.0, 3.0]])

    result = attribution.saliency(model, (first, second), target=0)

    assert isinstance(result.values, tuple)
    assert len(result.values) == 2
    first_values, second_values = result.values
    assert isinstance(first_values, Tensor)
    assert isinstance(second_values, Tensor)
    assert first_values.shape == first.shape
    assert second_values.shape == second.shape
    torch.testing.assert_close(first_values, torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(second_values, torch.tensor([[3.0, 0.0]]))


def test_integrated_gradients_two_input_shapes_and_completeness() -> None:
    """Multi-input Integrated Gradients satisfies completeness across leaves."""

    model = TwoInputSmooth()
    first = torch.tensor([[0.5, -0.7]], dtype=torch.float64)
    second = torch.tensor([[1.2, -0.3]], dtype=torch.float64)
    first_baseline = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    second_baseline = torch.tensor([[0.0, 0.4]], dtype=torch.float64)

    result = attribution.integrated_gradients(
        model,
        (first, second),
        target=lambda output: output.sum(),
        baseline=(first_baseline, second_baseline),
        n_steps=512,
    )

    assert isinstance(result.values, tuple)
    assert len(result.values) == 2
    first_values, second_values = result.values
    assert isinstance(first_values, Tensor)
    assert isinstance(second_values, Tensor)
    assert first_values.shape == first.shape
    assert second_values.shape == second.shape
    with torch.no_grad():
        target_delta = model(first, second).sum() - model(first_baseline, second_baseline).sum()
    attribution_sum = _sum_attribution_values(result.values)
    residual = (attribution_sum - target_delta).abs()

    assert residual <= 1e-4
    torch.testing.assert_close(attribution_sum, target_delta, rtol=1e-3, atol=1e-4)


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

    with pytest.raises(attribution.AttributionError, match="baseline must mirror"):
        attribution.integrated_gradients(
            model,
            (first, second),
            target=0,
            baseline=(torch.zeros_like(first),),
        )


def test_int_and_callable_targets_produce_valid_attributions() -> None:
    """Integer class targets and callable scalarizers both produce valid values."""

    model = TinyLinear(torch.tensor([[0.5, -0.25, 0.75], [1.0, 0.25, -0.5]]))
    inputs = torch.tensor([[0.3, -0.4, 0.8]])

    int_result = attribution.saliency(model, inputs, target=0)
    callable_result = attribution.saliency(
        model, inputs, target=lambda output: output[..., 1].sum()
    )

    assert int_result.values.shape == inputs.shape
    assert callable_result.values.shape == inputs.shape
    torch.testing.assert_close(int_result.values, torch.tensor([[0.5, 0.25, 0.75]]))
    torch.testing.assert_close(callable_result.values, torch.tensor([[1.0, 0.25, 0.5]]))
