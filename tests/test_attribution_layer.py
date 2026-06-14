"""Tests for native TorchLens layer attribution."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution


class TinyCnn(nn.Module):
    """Small CNN with deterministic layers for layer-attribution tests."""

    def __init__(self) -> None:
        """Initialize the convolutional and linear layers."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            conv = self.features[0]
            assert isinstance(conv, nn.Conv2d)
            conv.weight.copy_(
                torch.tensor(
                    [
                        [[[0.00, 0.25, 0.00], [0.25, 0.50, 0.25], [0.00, 0.25, 0.00]]],
                        [[[0.10, -0.20, 0.10], [-0.20, 0.40, -0.20], [0.10, -0.20, 0.10]]],
                    ]
                )
            )
            self.classifier.weight.copy_(torch.tensor([[0.75, -0.25], [-0.50, 1.00]]))

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the CNN.

        Parameters
        ----------
        inputs
            Input tensor with shape ``N, 1, H, W``.

        Returns
        -------
        Tensor
            Class logits with shape ``N, 2``.
        """

        features = self.features(inputs)
        pooled = features.mean(dim=(2, 3))
        return self.classifier(pooled)


class MultiInputCnn(nn.Module):
    """CNN that accepts multiple tensor inputs."""

    def __init__(self) -> None:
        """Initialize deterministic convolutional and linear layers."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            conv = self.features[0]
            assert isinstance(conv, nn.Conv2d)
            conv.weight.copy_(
                torch.tensor(
                    [
                        [[[0.10, 0.20, 0.10], [0.20, 0.40, 0.20], [0.10, 0.20, 0.10]]],
                        [[[0.00, -0.15, 0.00], [-0.15, 0.30, -0.15], [0.00, -0.15, 0.00]]],
                    ]
                )
            )
            self.classifier.weight.copy_(torch.tensor([[0.50, -0.25], [-0.10, 0.75]]))

    def forward(self, image: Tensor, residual: Tensor, *, gain: Tensor | None = None) -> Tensor:
        """Run the CNN on combined positional and keyword tensor inputs.

        Parameters
        ----------
        image
            Primary image tensor with shape ``N, 1, H, W``.
        residual
            Residual image tensor with shape ``N, 1, H, W``.
        gain
            Optional tensor multiplier broadcastable to image shape.

        Returns
        -------
        Tensor
            Class logits with shape ``N, 2``.
        """

        combined = image + residual
        if gain is not None:
            combined = combined * gain
        features = self.features(combined)
        pooled = features.mean(dim=(2, 3))
        return self.classifier(pooled)


class SmoothLayerMlp(nn.Module):
    """Smooth model with a named hidden layer for path-attribution tests."""

    def __init__(self) -> None:
        """Initialize deterministic double-precision layers."""

        super().__init__()
        self.hidden = nn.Linear(3, 4, bias=True, dtype=torch.float64)
        self.output = nn.Linear(4, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor(
                    [
                        [0.30, -0.20, 0.40],
                        [-0.10, 0.50, 0.20],
                        [0.60, 0.10, -0.30],
                        [-0.40, 0.25, 0.35],
                    ],
                    dtype=torch.float64,
                )
            )
            self.hidden.bias.copy_(torch.tensor([0.05, -0.10, 0.20, -0.15], dtype=torch.float64))
            self.output.weight.copy_(
                torch.tensor(
                    [[0.40, -0.30, 0.20, 0.10], [-0.25, 0.35, 0.15, -0.45]], dtype=torch.float64
                )
            )
            self.output.bias.copy_(torch.tensor([0.02, -0.03], dtype=torch.float64))

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the model.

        Parameters
        ----------
        inputs
            Input tensor with shape ``N, 3``.

        Returns
        -------
        Tensor
            Output tensor with shape ``N, 2``.
        """

        hidden = self.hidden(inputs)
        return self.output(torch.tanh(hidden))


class MultiInputSmoothLayerMlp(nn.Module):
    """Smooth named-layer model that accepts multiple positional tensor inputs."""

    def __init__(self) -> None:
        """Initialize deterministic double-precision layers."""

        super().__init__()
        self.hidden = nn.Linear(2, 3, bias=True, dtype=torch.float64)
        self.output = nn.Linear(3, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor(
                    [[0.25, -0.35], [0.45, 0.15], [-0.20, 0.55]],
                    dtype=torch.float64,
                )
            )
            self.hidden.bias.copy_(torch.tensor([0.10, -0.05, 0.15], dtype=torch.float64))
            self.output.weight.copy_(torch.tensor([[0.60, -0.40, 0.30]], dtype=torch.float64))

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        """Run the model on two positional tensor inputs.

        Parameters
        ----------
        first
            First input tensor with shape ``N, 2``.
        second
            Second input tensor with shape ``N, 2``.

        Returns
        -------
        Tensor
            Output tensor with shape ``N, 1``.
        """

        hidden = self.hidden(first + 0.5 * second)
        return self.output(torch.tanh(hidden))


def test_grad_cam_upsamples_to_input_spatial_size_and_is_finite() -> None:
    """Grad-CAM returns a finite non-negative map at input spatial resolution."""

    model = TinyCnn()
    inputs = torch.linspace(-1.0, 1.0, steps=25).reshape(1, 1, 5, 5)

    result = attribution.grad_cam(model, inputs, target=1, layer="features.0")

    assert result.method == "grad_cam"
    assert result.values.shape == (1, 1, 5, 5)
    assert result.extra == {"layer": "features.0", "relu": True}
    assert torch.isfinite(result.values).all()
    assert (result.values >= 0).all()


def test_grad_cam_callable_target_form() -> None:
    """Grad-CAM accepts a callable output scalarizer."""

    model = TinyCnn()
    inputs = torch.randn(2, 1, 6, 6)

    result = attribution.grad_cam(
        model,
        inputs,
        target=lambda output: output[:, 0].sum(),
        layer="features.0",
        relu=False,
    )

    assert result.values.shape == (2, 1, 6, 6)
    assert torch.isfinite(result.values).all()
    assert result.extra == {"layer": "features.0", "relu": False}


def test_grad_cam_rejects_non_conv_style_layer() -> None:
    """Grad-CAM rejects layers that do not emit N,C,H,W feature maps."""

    model = TinyCnn()
    inputs = torch.randn(1, 1, 5, 5)

    with pytest.raises(attribution.AttributionError, match="4D N,C,H,W feature map"):
        attribution.grad_cam(model, inputs, target=0, layer="classifier")


def test_layer_attribution_activation_x_grad_shape_matches_layer_activation() -> None:
    """Activation-times-gradient has the same shape as the target activation."""

    model = TinyCnn()
    inputs = torch.randn(1, 1, 5, 5)

    result = attribution.layer_attribution(
        model,
        inputs,
        target=0,
        layer="features.0",
        method="activation_x_grad",
    )

    assert result.method == "layer_activation_x_grad"
    assert result.values.shape == (1, 2, 5, 5)
    assert result.extra == {"layer": "features.0"}
    assert torch.isfinite(result.values).all()


def test_layer_attribution_grad_method_callable_target() -> None:
    """Gradient layer attribution returns absolute target gradients."""

    model = TinyCnn()
    inputs = torch.randn(2, 1, 4, 4)

    result = attribution.layer_attribution(
        model,
        inputs,
        target=lambda output: output[:, 1].sum(),
        layer="features.0",
        method="grad",
    )

    assert result.method == "layer_grad"
    assert result.values.shape == (2, 2, 4, 4)
    assert torch.isfinite(result.values).all()
    assert (result.values >= 0).all()


def test_bad_layer_name_lists_available_options() -> None:
    """Missing layer errors include available conv-like layer suggestions."""

    model = TinyCnn()
    inputs = torch.randn(1, 1, 5, 5)

    with pytest.raises(
        attribution.AttributionError,
        match="layer 'missing' was not found; available conv-like layers include: features.0",
    ):
        attribution.grad_cam(model, inputs, target=0, layer="missing")


def test_grad_cam_supports_multi_input_model_call() -> None:
    """Grad-CAM threads multiple positional tensors through the model call."""

    model = MultiInputCnn()
    image = torch.randn(1, 1, 5, 5)
    residual = torch.randn(1, 1, 5, 5)

    result = attribution.grad_cam(model, (image, residual), target=0, layer="features.0")

    assert result.method == "grad_cam"
    assert result.values.shape == (1, 1, 5, 5)
    assert torch.isfinite(result.values).all()


def test_layer_attribution_supports_tensor_kwargs_model_call() -> None:
    """Layer attribution threads tensor kwargs through the model call."""

    model = MultiInputCnn()
    image = torch.randn(1, 1, 5, 5)
    residual = torch.randn(1, 1, 5, 5)
    gain = torch.full((1, 1, 5, 5), 0.5)

    result = attribution.layer_attribution(
        model,
        (image, residual),
        {"gain": gain},
        target=1,
        layer="features.0",
    )

    assert result.method == "layer_activation_x_grad"
    assert result.values.shape == (1, 2, 5, 5)
    assert torch.isfinite(result.values).all()


def test_layer_integrated_gradients_completeness_and_shape() -> None:
    """Layer Integrated Gradients sums to the target delta on a smooth path."""

    model = SmoothLayerMlp()
    inputs = torch.tensor([[0.60, -0.20, 0.40]], dtype=torch.float64)
    baseline = torch.tensor([[-0.10, 0.05, 0.20]], dtype=torch.float64)

    result = attribution.layer_integrated_gradients(
        model,
        inputs,
        target=1,
        layer="hidden",
        baseline=baseline,
        n_steps=512,
    )

    with torch.no_grad():
        target_delta = model(inputs)[..., 1].sum() - model(baseline)[..., 1].sum()
        expected_shape = model.hidden(inputs).shape
    attribution_sum = result.values.sum()

    assert result.method == "layer_integrated_gradients"
    assert result.extra == {"layer": "hidden", "n_steps": 512}
    assert result.values.shape == expected_shape
    assert torch.isfinite(result.values).all()
    torch.testing.assert_close(attribution_sum, target_delta, rtol=1e-3, atol=1e-4)


def test_layer_conductance_completeness_and_shape() -> None:
    """Layer Conductance sums to the target delta on a smooth path."""

    model = SmoothLayerMlp()
    inputs = torch.tensor([[0.45, -0.35, 0.25]], dtype=torch.float64)
    baseline = torch.tensor([[-0.15, 0.10, 0.05]], dtype=torch.float64)

    result = attribution.layer_conductance(
        model,
        inputs,
        target=lambda output: output[..., 0].sum(),
        layer="hidden",
        baseline=baseline,
        n_steps=512,
    )

    with torch.no_grad():
        target_delta = model(inputs)[..., 0].sum() - model(baseline)[..., 0].sum()
        expected_shape = model.hidden(inputs).shape
    attribution_sum = result.values.sum()

    assert result.method == "layer_conductance"
    assert result.extra == {"layer": "hidden", "n_steps": 512}
    assert result.values.shape == expected_shape
    assert torch.isfinite(result.values).all()
    torch.testing.assert_close(attribution_sum, target_delta, rtol=1e-3, atol=1e-4)


def test_layer_path_methods_reject_bad_layer_name() -> None:
    """Layer path methods reuse named-layer validation errors."""

    model = SmoothLayerMlp()
    inputs = torch.randn(1, 3, dtype=torch.float64)

    with pytest.raises(
        attribution.AttributionError,
        match="layer 'missing' was not found; available conv-like layers include: hidden",
    ):
        attribution.layer_integrated_gradients(
            model,
            inputs,
            target=0,
            layer="missing",
        )


def test_layer_path_methods_support_multi_input_model_call() -> None:
    """Layer path methods thread multiple positional tensors through the model call."""

    model = MultiInputSmoothLayerMlp()
    first = torch.tensor([[0.30, -0.60]], dtype=torch.float64)
    second = torch.tensor([[0.20, 0.40]], dtype=torch.float64)
    first_baseline = torch.tensor([[0.05, -0.10]], dtype=torch.float64)
    second_baseline = torch.tensor([[-0.15, 0.05]], dtype=torch.float64)

    lig_result = attribution.layer_integrated_gradients(
        model,
        (first, second),
        target=lambda output: output.sum(),
        layer="hidden",
        baseline=(first_baseline, second_baseline),
        n_steps=512,
    )
    conductance_result = attribution.layer_conductance(
        model,
        (first, second),
        target=lambda output: output.sum(),
        layer="hidden",
        baseline=(first_baseline, second_baseline),
        n_steps=512,
    )

    with torch.no_grad():
        target_delta = model(first, second).sum() - model(first_baseline, second_baseline).sum()
        expected_shape = model.hidden(first + 0.5 * second).shape

    assert lig_result.values.shape == expected_shape
    assert conductance_result.values.shape == expected_shape
    torch.testing.assert_close(lig_result.values.sum(), target_delta, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(
        conductance_result.values.sum(),
        target_delta,
        rtol=1e-3,
        atol=1e-4,
    )
