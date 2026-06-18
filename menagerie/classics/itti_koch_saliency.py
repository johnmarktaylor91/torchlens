"""Itti-Koch-Niebur saliency model, 1998, Laurent Itti, Christof Koch, Ernst Niebur.

Paper: A model of saliency-based visual attention for rapid scene analysis.
Feature pyramids compare center and surround scales for intensity, color, and
orientation conspicuity maps, which are normalized and summed into saliency.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [("Itti-Koch-Niebur Saliency Model", "build", "example_input", "1998", "DC")]


class IttiKochNieburSaliency(nn.Module):
    """Traceable center-surround saliency module."""

    def __init__(self) -> None:
        """Initialize fixed Gaussian and oriented filters."""
        super().__init__()
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("kernel_h", kernel.view(1, 1, 1, 5))
        self.register_buffer("kernel_v", kernel.view(1, 1, 5, 1))
        self.register_buffer("gabor", self._make_oriented_filters())

    def _make_oriented_filters(self) -> Tensor:
        """Create four fixed orientation filters for the conspicuity map.

        Returns
        -------
        Tensor
            Filter tensor with shape ``(4, 1, 9, 9)``.
        """
        coords = torch.linspace(-1.0, 1.0, 9)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        filters = []
        for idx in range(4):
            theta = math.pi * float(idx) / 4.0
            xr = xx * math.cos(theta) + yy * math.sin(theta)
            yr = -xx * math.sin(theta) + yy * math.cos(theta)
            filt = torch.exp(-(xr.square() + yr.square()) / 0.35) * torch.cos(3.0 * math.pi * xr)
            filters.append(filt - filt.mean())
        return torch.stack(filters).unsqueeze(1)

    def _blur(self, x: Tensor) -> Tensor:
        """Apply a separable Gaussian blur channelwise.

        Parameters
        ----------
        x
            Tensor with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Blurred tensor.
        """
        channels = x.shape[1]
        weight_h = self.kernel_h.expand(channels, 1, 1, 5)
        weight_v = self.kernel_v.expand(channels, 1, 5, 1)
        x = F.conv2d(x, weight_h, padding=(0, 2), groups=channels)
        return F.conv2d(x, weight_v, padding=(2, 0), groups=channels)

    def _normalize_map(self, x: Tensor) -> Tensor:
        """Normalize a conspicuity map by mean-centered contrast.

        Parameters
        ----------
        x
            Conspicuity tensor.

        Returns
        -------
        Tensor
            Nonnegative normalized map.
        """
        x = x - x.amin(dim=(-2, -1), keepdim=True)
        denom = x.amax(dim=(-2, -1), keepdim=True) + 1.0e-6
        scaled = x / denom
        return (
            scaled
            * (
                scaled.amax(dim=(-2, -1), keepdim=True) - scaled.mean(dim=(-2, -1), keepdim=True)
            ).square()
        )

    def _center_surround(self, x: Tensor) -> Tensor:
        """Compare fine-scale centers against blurred surrounds.

        Parameters
        ----------
        x
            Feature tensor.

        Returns
        -------
        Tensor
            Center-surround contrast map.
        """
        surround = self._blur(self._blur(x))
        return torch.abs(x - surround)

    def forward(self, x: Tensor) -> Tensor:
        """Compute a bottom-up saliency map.

        Parameters
        ----------
        x
            RGB image tensor with shape ``(B, 3, 256, 256)``.

        Returns
        -------
        Tensor
            Saliency tensor with shape ``(B, 1, 256, 256)``.
        """
        intensity = x.mean(dim=1, keepdim=True)
        red_green = x[:, 0:1] - x[:, 1:2]
        blue_yellow = x[:, 2:3] - 0.5 * (x[:, 0:1] + x[:, 1:2])
        orientation = torch.relu(F.conv2d(intensity, self.gabor, padding=4)).mean(
            dim=1, keepdim=True
        )
        maps = [
            self._normalize_map(self._center_surround(intensity)),
            self._normalize_map(self._center_surround(red_green)),
            self._normalize_map(self._center_surround(blue_yellow)),
            self._normalize_map(self._center_surround(orientation)),
        ]
        return torch.stack(maps, dim=0).sum(dim=0) / 4.0


def build() -> nn.Module:
    """Build a compact Itti-Koch-Niebur saliency module.

    Returns
    -------
    nn.Module
        Fixed-filter saliency module.
    """
    return IttiKochNieburSaliency()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 256, 256)``.
    """
    return torch.randn(1, 3, 256, 256)
