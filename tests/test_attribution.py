"""Tests for native TorchLens input attribution."""

from __future__ import annotations

import torch
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


def test_v1_rejects_multi_input_and_pytree_inputs() -> None:
    """Tuple and dict inputs raise the v1 unsupported-input error."""

    model = TinyLinear(torch.tensor([[1.0, 2.0]]))
    first = torch.ones(1, 2)
    second = torch.zeros(1, 2)
    expected = "multi-input / kwarg / pytree inputs unsupported in v1; pass a single tensor"

    for bad_inputs in [(first, second), {"inputs": first}]:
        try:
            attribution.saliency(model, bad_inputs, target=0)  # type: ignore[arg-type]
        except attribution.AttributionError as exc:
            assert str(exc) == expected
        else:
            raise AssertionError("AttributionError was not raised")


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
