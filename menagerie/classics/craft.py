"""CRAFT: Character Region Awareness for scene text detection.

Paper: Baek et al. 2019, "Character Region Awareness for Text Detection"
(CVPR), arXiv:1904.01941.

The compact reconstruction follows the published CRAFT structure: a VGG-like
feature extractor, U-Net-style feature fusion, and two heatmap heads for
character-region and character-affinity scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CraftBlock(nn.Module):
    """Convolution, normalization, and ReLU block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize one CRAFT convolution block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Transformed feature map.
        """

        return self.net(x)


class CRAFT(nn.Module):
    """Compact CRAFT detector with region and affinity heads."""

    def __init__(self) -> None:
        """Initialize VGG-like encoder, fusion path, and heatmap heads."""

        super().__init__()
        self.enc1 = CraftBlock(3, 16)
        self.enc2 = CraftBlock(16, 32)
        self.enc3 = CraftBlock(32, 64)
        self.fuse2 = CraftBlock(96, 32)
        self.fuse1 = CraftBlock(48, 24)
        self.region = nn.Conv2d(24, 1, 1)
        self.affinity = nn.Conv2d(24, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict character region and affinity heatmaps.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Two-channel heatmap tensor.
        """

        c1 = self.enc1(image)
        c2 = self.enc2(F.max_pool2d(c1, 2))
        c3 = self.enc3(F.max_pool2d(c2, 2))
        u2 = F.interpolate(c3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        f2 = self.fuse2(torch.cat([u2, c2], dim=1))
        u1 = F.interpolate(f2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse1(torch.cat([u1, c1], dim=1))
        return torch.cat(
            [torch.sigmoid(self.region(fused)), torch.sigmoid(self.affinity(fused))], dim=1
        )


def build() -> nn.Module:
    """Build the compact CRAFT detector.

    Returns
    -------
    nn.Module
        Random-initialized CRAFT module.
    """

    return CRAFT()


def example_input() -> torch.Tensor:
    """Create a small RGB text-line image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 64)``.
    """

    return torch.randn(1, 3, 32, 64)


MENAGERIE_ENTRIES = [
    ("CRAFT", "build", "example_input", "2019", "E5"),
]
