# FAITHFUL REIMPLEMENTATION from arXiv:1910.13236 (no public code) -- A/B codex
"""Deep Horizon CNN pair from the paper description."""

from __future__ import annotations

import torch
from torch import nn


class ConvUnit(nn.Module):
    """Convolutional unit shown in Fig. 1 of the Deep Horizon paper."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        """Create a convolution, max-pooling, and dropout unit.

        Parameters
        ----------
        in_channels:
            Number of input image channels.
        out_channels:
            Number of convolution filters.
        dropout:
            Dropout probability used for Monte Carlo dropout.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional unit.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Downsampled feature maps.
        """
        return self.net(x)


class DenseUnit(nn.Module):
    """Dense unit shown in Fig. 1 of the Deep Horizon paper."""

    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        """Create a dense, ReLU, and dropout unit.

        Parameters
        ----------
        in_features:
            Number of input features.
        out_features:
            Number of dense neurons.
        dropout:
            Dropout probability used for Monte Carlo dropout.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dense unit.

        Parameters
        ----------
        x:
            Flattened feature tensor.

        Returns
        -------
        torch.Tensor
            Dense features.
        """
        return self.net(x)


class RegressionArm(nn.Module):
    """One parameter-specific Network I regression arm."""

    def __init__(self, in_features: int, dropout: float) -> None:
        """Create the dense arm and its prediction/uncertainty heads.

        Parameters
        ----------
        in_features:
            Number of flattened CNN features.
        dropout:
            Dropout probability used for Monte Carlo dropout.
        """
        super().__init__()
        self.dense64 = DenseUnit(in_features, 64, dropout)
        self.dense32 = DenseUnit(64, 32, dropout)
        self.shared16 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.prediction = nn.Linear(16, 1)
        self.uncertainty = nn.Sequential(
            nn.Linear(16, 1),
            nn.Tanh(),
            nn.Linear(1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one physical parameter and its aleatoric uncertainty.

        Parameters
        ----------
        x:
            Flattened CNN features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Parameter prediction and positive aleatoric uncertainty.
        """
        hidden = self.shared16(self.dense32(self.dense64(x)))
        prediction = self.prediction(hidden)
        aleatoric = torch.exp(self.uncertainty(hidden))
        return prediction, aleatoric


class DeepHorizon(nn.Module):
    """Two-CNN Deep Horizon model: Bayesian regressor plus spin classifier."""

    def __init__(self, image_size: int = 64, dropout: float = 0.01) -> None:
        """Create Network I and Network II from Fig. 1.

        Parameters
        ----------
        image_size:
            Square image side length.
        dropout:
            Dropout rate; the paper reports 0.01 after tuning.
        """
        super().__init__()
        flattened = 64 * (image_size // 8) * (image_size // 8)
        self.regression_features = nn.Sequential(
            ConvUnit(1, 16, dropout),
            ConvUnit(16, 32, dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier_features = nn.Sequential(
            ConvUnit(1, 16, dropout),
            ConvUnit(16, 32, dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.arms = nn.ModuleList(RegressionArm(flattened, dropout) for _ in range(5))
        self.spin = nn.Sequential(nn.Linear(flattened, 5), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the regression and spin-classification CNNs.

        Parameters
        ----------
        x:
            Single-channel black-hole image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Five regression predictions, five aleatoric uncertainties, and five spin probabilities.
        """
        reg_features = self.regression_features(x)
        predictions = []
        uncertainties = []
        for arm in self.arms:
            prediction, uncertainty = arm(reg_features)
            predictions.append(prediction)
            uncertainties.append(uncertainty)
        spin_probs = self.spin(self.classifier_features(x))
        return torch.cat(predictions, dim=-1), torch.cat(uncertainties, dim=-1), spin_probs


def build_deep_horizon() -> DeepHorizon:
    """Build a tiny-input Deep Horizon model.

    Returns
    -------
    DeepHorizon
        The reimplemented model.
    """
    return DeepHorizon(image_size=64)


def example_input_deep_horizon() -> torch.Tensor:
    """Create an example black-hole image tensor.

    Returns
    -------
    torch.Tensor
        Example image of shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("Deep Horizon", "build_deep_horizon", "example_input_deep_horizon", 2020, "REIMPL")
]
