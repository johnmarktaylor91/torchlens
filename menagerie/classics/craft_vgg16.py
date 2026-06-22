"""CRAFT text detector with a VGG-style encoder and U-Net decoder.

Baek et al., CVPR 2019, "Character Region Awareness for Text Detection." CRAFT
uses VGG-16-like convolutional features, U-Net-style feature aggregation, and two
output score maps: character region and character affinity.  This compact
random-init reconstruction keeps the VGG/skip/decoder/two-map structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactCRAFT(nn.Module):
    """Compact CRAFT detector."""

    def __init__(self) -> None:
        """Initialize encoder, decoder, and score head."""

        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU())
        self.score = nn.Conv2d(16, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict region and affinity score maps.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        torch.Tensor
            Two-channel score map.
        """

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        u2 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, f2], dim=1))
        u1 = F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, f1], dim=1))
        return torch.sigmoid(self.score(d1))


def build() -> nn.Module:
    """Build compact CRAFT.

    Returns
    -------
    nn.Module
        Random-init CRAFT reconstruction.
    """

    return CompactCRAFT()


def example_input() -> torch.Tensor:
    """Create image input.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 48, 48)


MENAGERIE_ENTRIES = [("CRAFT-VGG16", "build", "example_input", "2019", "CV")]
