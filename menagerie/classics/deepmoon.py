"""DeepMoon compact U-Net crater detector.

Silburt et al., "Lunar Crater Identification via Deep Learning", 2019.
DeepMoon uses a U-shaped fully convolutional network on lunar digital elevation
maps to predict crater masks, followed by geometric post-processing for crater
centers and radii.  The distinctive architecture is the DEM-to-mask U-Net, not a
generic image classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    """Downsampling block for DeepMoon U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolutions.

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
        """Apply down block convolutions.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Feature tensor.
        """

        return F.relu(self.conv2(F.relu(self.conv1(x))))


class DeepMoonCompact(nn.Module):
    """Compact DEM U-Net for crater mask prediction."""

    def __init__(self, width: int = 16) -> None:
        """Initialize encoder, decoder, and crater-mask head.

        Parameters
        ----------
        width:
            Base feature width.
        """

        super().__init__()
        self.d1 = Down(1, width)
        self.d2 = Down(width, width * 2)
        self.d3 = Down(width * 2, width * 4)
        self.u2 = Down(width * 6, width * 2)
        self.u1 = Down(width * 3, width)
        self.mask = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict crater mask logits from a lunar DEM patch.

        Parameters
        ----------
        x:
            DEM tensor ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Crater mask logits.
        """

        e1 = self.d1(x)
        e2 = self.d2(F.max_pool2d(e1, 2))
        e3 = self.d3(F.max_pool2d(e2, 2))
        y = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u2(torch.cat([y, e2], dim=1))
        y = F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.u1(torch.cat([y, e1], dim=1))
        return self.mask(y)


def build() -> nn.Module:
    """Build compact DeepMoon.

    Returns
    -------
    nn.Module
        DeepMoon crater detector.
    """

    return DeepMoonCompact()


def example_input() -> torch.Tensor:
    """Create a lunar DEM image patch.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 1, 64, 64)``.
    """

    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("DeepMoon (U-Net lunar DEM crater mask detector)", "build", "example_input", "2019", "DC"),
]
