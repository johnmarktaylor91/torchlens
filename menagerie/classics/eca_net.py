"""ECA-Net: Efficient Channel Attention.

Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural
Networks. Wang et al., CVPR 2020.

ECA keeps SENet-style global channel descriptors but removes dimensionality
reduction, using a lightweight 1D convolution to model local cross-channel
interactions before sigmoid channel reweighting.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ECALayer(nn.Module):
    """Efficient Channel Attention layer using descriptor-space 1D convolution."""

    def __init__(self, kernel_size: int = 3) -> None:
        """Initialize the channel-interaction convolution.

        Parameters
        ----------
        kernel_size:
            Odd 1D convolution kernel size over channel descriptors.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ECA channel reweighting.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Channel-reweighted feature map.
        """
        descriptor = self.pool(x).flatten(2).transpose(1, 2)
        weights = torch.sigmoid(self.conv(descriptor)).transpose(1, 2).unsqueeze(-1)
        return x * weights


class ECADemoNet(nn.Module):
    """Compact CNN with ECA inserted after each convolutional stage."""

    def __init__(self, channels: int = 16, num_classes: int = 5) -> None:
        """Initialize the demonstration classifier.

        Parameters
        ----------
        channels:
            Width of the convolutional trunk.
        num_classes:
            Number of output logits.
        """
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            ECALayer(kernel_size=3),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            ECALayer(kernel_size=3),
        )
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits through ECA-refined stages.

        Parameters
        ----------
        x:
            RGB image tensor with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = self.stage2(self.stage1(x))
        x = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return self.classifier(x)


def build() -> nn.Module:
    """Build a compact ECA-Net demonstration network.

    Returns
    -------
    nn.Module
        Random-initialized ECA demo network.
    """
    return ECADemoNet()


def example_input() -> Tensor:
    """Return a small traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("ECA-Net (Efficient Channel Attention)", "build", "example_input", "2020", "DC")
]
