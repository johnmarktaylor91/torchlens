"""Tests for native TorchLens layer attribution."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens
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
    assert result.extra["layer"] == "features.0"
    assert result.extra["relu"] is False


def test_grad_cam_rejects_non_conv_style_layer() -> None:
    """Grad-CAM rejects layers that do not emit N,C,H,W feature maps."""

    model = TinyCnn()
    inputs = torch.randn(1, 1, 5, 5)

    with pytest.raises(torchlens.errors.AttributionError, match="4D N,C,H,W feature map"):
        attribution.grad_cam(model, inputs, target=0, layer="classifier")


def test_bad_layer_name_lists_available_options() -> None:
    """Missing layer errors include available conv-like layer suggestions."""

    model = TinyCnn()
    inputs = torch.randn(1, 1, 5, 5)

    with pytest.raises(
        torchlens.errors.AttributionError,
        match="layer 'missing' was not found; available conv-like layers include: features.0",
    ):
        attribution.grad_cam(model, inputs, target=0, layer="missing")


def test_layer_path_methods_reject_bad_layer_name() -> None:
    """Layer path methods reuse named-layer validation errors."""

    model = SmoothLayerMlp()
    inputs = torch.randn(1, 3, dtype=torch.float64)

    with pytest.raises(
        torchlens.errors.AttributionError,
        match="layer 'missing' was not found; available conv-like layers include: hidden",
    ):
        attribution.layer_integrated_gradients(
            model,
            inputs,
            target=0,
            layer="missing",
        )
