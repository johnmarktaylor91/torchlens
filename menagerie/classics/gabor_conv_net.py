"""Gabor Convolutional Network, 2016, Javier Luan et al.

Paper: Gabor Convolutional Networks.
The first convolutional layer is generated from trainable Gabor parameters
before standard convolutional blocks classify the image.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Gabor Convolutional Network (learnable Gabor-CNN)", "build", "example_input", "2016", "DC")
]


class GaborConvolutionalNetwork(nn.Module):
    """Small CNN with a parameterized Gabor first layer."""

    def __init__(self, filters: int = 8, num_classes: int = 10) -> None:
        """Initialize Gabor parameters and downstream classifier.

        Parameters
        ----------
        filters
            Number of generated Gabor filters.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, filters))
        self.frequency = nn.Parameter(torch.full((filters,), 2.5))
        self.sigma = nn.Parameter(torch.full((filters,), 0.6))
        self.phase = nn.Parameter(torch.zeros(filters))
        self.mix = nn.Conv2d(filters, 16, kernel_size=3, padding=1)
        self.classifier = nn.Linear(16 * 8 * 8, num_classes)
        coords = torch.linspace(-1.0, 1.0, 7)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)

    def _gabor_weight(self) -> Tensor:
        """Generate first-layer Gabor filters from trainable parameters.

        Returns
        -------
        Tensor
            Filter tensor with shape ``(F, 3, 7, 7)``.
        """
        theta = self.theta.view(-1, 1, 1)
        sigma = torch.relu(self.sigma).view(-1, 1, 1) + 0.05
        frequency = torch.relu(self.frequency).view(-1, 1, 1)
        phase = self.phase.view(-1, 1, 1)
        xr = self.xx * torch.cos(theta) + self.yy * torch.sin(theta)
        yr = -self.xx * torch.sin(theta) + self.yy * torch.cos(theta)
        envelope = torch.exp(-(xr.square() + yr.square()) / (2.0 * sigma.square()))
        carrier = torch.cos(frequency * math.pi * xr + phase)
        filters = envelope * carrier
        filters = filters - filters.mean(dim=(-2, -1), keepdim=True)
        return filters.unsqueeze(1).expand(-1, 3, -1, -1) / 3.0

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image with generated Gabor filters.

        Parameters
        ----------
        x
            RGB image tensor with shape ``(B, 3, 32, 32)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = torch.relu(F.conv2d(x, self._gabor_weight(), padding=3))
        x = F.max_pool2d(torch.relu(self.mix(x)), 2)
        x = F.max_pool2d(x, 2)
        return self.classifier(x.flatten(1))


def build() -> nn.Module:
    """Build a compact Gabor-CNN.

    Returns
    -------
    nn.Module
        Random-initialized Gabor-CNN classifier.
    """
    return GaborConvolutionalNetwork()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)
