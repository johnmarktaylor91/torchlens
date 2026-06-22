"""StarDist compact star-convex instance segmentation network.

Schmidt et al., "Cell Detection with Star-convex Polygons", MICCAI 2018, and
Weigert et al., "Star-convex Polyhedra for 3D Object Detection and Segmentation
in Microscopy", WACV 2020.  StarDist densely predicts an object probability and
radial distances along fixed rays; high-probability pixels vote for star-convex
object polygons/polyhedra followed by non-maximum suppression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block for the compact StarDist U-Net backbone."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize two same-resolution convolutions.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Refined feature tensor.
        """

        return F.relu(self.conv2(F.relu(self.conv1(x))))


class StarDistCompact(nn.Module):
    """Small U-Net that predicts objectness and ray distances per pixel."""

    def __init__(self, in_channels: int = 1, rays: int = 16, width: int = 16) -> None:
        """Initialize the compact StarDist network.

        Parameters
        ----------
        in_channels:
            Input image channels.
        rays:
            Number of fixed star-convex radial directions.
        width:
            Base feature width.
        """

        super().__init__()
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.bottleneck = ConvBlock(width * 2, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.prob_head = nn.Conv2d(width, 1, 1)
        self.dist_head = nn.Conv2d(width, rays, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel object logit and positive radial distances.

        Parameters
        ----------
        x:
            Image tensor ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Concatenated ``(prob_logit, ray_distances...)`` maps.
        """

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.cat([self.prob_head(d1), F.softplus(self.dist_head(d1))], dim=1)


def build() -> nn.Module:
    """Build a compact random-init StarDist model.

    Returns
    -------
    nn.Module
        StarDist-style U-Net.
    """

    return StarDistCompact()


def example_input() -> torch.Tensor:
    """Create a small grayscale microscopy input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 1, 32, 32)``.
    """

    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "StarDist (star-convex polygon distance-field instance segmenter)",
        "build",
        "example_input",
        "2018",
        "DC",
    ),
]
