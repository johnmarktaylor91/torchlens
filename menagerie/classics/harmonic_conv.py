"""Harmonic Networks with circular-harmonic filters, 2017.

Paper: Worrall et al. 2017, "Harmonic Networks: Deep Translation and
Rotation Equivariance". This compact H-Net sketch builds steerable cosine and
sine filter pairs and applies a magnitude nonlinearity over rotation-order
streams, omitting full phase alignment and multi-layer order bookkeeping.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HarmonicConv2d(nn.Module):
    """Circular-harmonic convolution layer returning magnitudes."""

    def __init__(self, in_channels: int = 2, out_channels: int = 4, kernel_size: int = 5) -> None:
        """Initialize radial envelopes and harmonic-order coefficients.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output streams.
        kernel_size
            Odd spatial kernel size.
        """
        super().__init__()
        self.coeff = nn.Parameter(torch.randn(out_channels, in_channels, 3) * 0.1)
        grid = torch.linspace(-1.0, 1.0, kernel_size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy).clamp_min(1e-5)
        angle = torch.atan2(yy, xx)
        envelope = torch.exp(-3.0 * radius * radius)
        self.register_buffer("angle", angle)
        self.register_buffer("envelope", envelope)

    def forward(self, x: Tensor) -> Tensor:
        """Apply harmonic filters and return magnitude streams.

        Parameters
        ----------
        x
            Image tensor with shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Magnitude activation tensor.
        """
        filters_re: list[Tensor] = []
        filters_im: list[Tensor] = []
        for order in range(3):
            phase = float(order) * self.angle
            basis_re = torch.cos(phase) * self.envelope
            basis_im = torch.sin(phase) * self.envelope
            filters_re.append(self.coeff[:, :, order, None, None] * basis_re)
            filters_im.append(self.coeff[:, :, order, None, None] * basis_im)
        weight_re = torch.stack(filters_re, dim=0).sum(dim=0)
        weight_im = torch.stack(filters_im, dim=0).sum(dim=0)
        pad = self.angle.shape[0] // 2
        response_re = F.conv2d(x, weight_re, padding=pad)
        response_im = F.conv2d(x, weight_im, padding=pad)
        return torch.sqrt(response_re * response_re + response_im * response_im + 1e-6)


class HarmonicNetwork(nn.Module):
    """Small H-Net stack with harmonic convolution and readout."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3) -> None:
        """Initialize the harmonic convolutional classifier.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output units.
        """
        super().__init__()
        self.hconv = HarmonicConv2d(in_channels, 4)
        self.readout = nn.Linear(4, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Compute harmonic magnitudes and global average readout.

        Parameters
        ----------
        x
            Image tensor with shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Output tensor.
        """
        magnitude = torch.relu(self.hconv(x))
        pooled = magnitude.mean(dim=(2, 3))
        return self.readout(pooled)


def build() -> nn.Module:
    """Build a small harmonic network.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return HarmonicNetwork()


def example_input() -> Tensor:
    """Return an example two-channel image.

    Returns
    -------
    Tensor
        Example tensor.
    """
    y = torch.linspace(-1.0, 1.0, 16)
    x = torch.linspace(-1.0, 1.0, 16)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    first = torch.sin(math.pi * xx) * torch.cos(math.pi * yy)
    second = torch.cos(math.pi * xx * yy)
    return torch.stack((first, second), dim=0).unsqueeze(0)


MENAGERIE_ENTRIES = [("Harmonic Network (H-Net)", "build", "example_input", "2017", "CH-C")]
