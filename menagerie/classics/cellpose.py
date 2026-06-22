"""Cellpose compact U-Net with style-conditioned flow and cell-probability heads.

Stringer et al., "Cellpose: a generalist algorithm for cellular segmentation",
Nature Methods 2021.  Cellpose is not a generic semantic segmenter: the network is
a residual U-Net that predicts two spatial flow components plus a cell-probability
logit, and older Cellpose variants also expose a global style vector from the
encoder.  Masks are recovered by running pixel dynamics along the predicted flow
field.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block used in the compact Cellpose U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        """

        super().__init__()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolutional refinement.

        Parameters
        ----------
        x:
            Image feature tensor.

        Returns
        -------
        torch.Tensor
            Refined features.
        """

        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.proj(x))


class CellposeCompact(nn.Module):
    """Small Cellpose-style residual U-Net with style modulation and flow heads."""

    def __init__(self, in_channels: int = 2, width: int = 16, style_dim: int = 16) -> None:
        """Initialize encoder, decoder, style, and three-channel output head.

        Parameters
        ----------
        in_channels:
            Microscopy input channels.
        width:
            Base feature width.
        style_dim:
            Global style-vector size.
        """

        super().__init__()
        self.enc1 = ResidualBlock(in_channels, width)
        self.enc2 = ResidualBlock(width, width * 2)
        self.enc3 = ResidualBlock(width * 2, width * 4)
        self.style = nn.Linear(width * 4, style_dim)
        self.style_to_bottleneck = nn.Linear(style_dim, width * 4)
        self.dec2 = ResidualBlock(width * 6, width * 2)
        self.dec1 = ResidualBlock(width * 3, width)
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Cellpose ``dy``, ``dx``, and cell-probability logits.

        Parameters
        ----------
        x:
            Microscopy image tensor ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Three output maps: vertical flow, horizontal flow, and cell probability.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        style = F.normalize(self.style(e3.mean(dim=(2, 3))), dim=-1)
        gain = torch.tanh(self.style_to_bottleneck(style)).unsqueeze(-1).unsqueeze(-1)
        e3 = e3 * (1.0 + gain)
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def build() -> nn.Module:
    """Build a compact random-init Cellpose model.

    Returns
    -------
    nn.Module
        Cellpose-style residual U-Net.
    """

    return CellposeCompact()


def example_input() -> torch.Tensor:
    """Create a small two-channel microscopy input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 2, 32, 32)``.
    """

    return torch.randn(1, 2, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Cellpose (style-conditioned U-Net flow-field instance segmenter)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
