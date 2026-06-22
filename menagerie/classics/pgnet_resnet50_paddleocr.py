"""PGNet-ResNet50-PaddleOCR: point-gathering arbitrarily-shaped text spotting.

Paper: Wang et al. 2021, "PGNet: Real-time Arbitrarily-Shaped Text Spotting
with Point Gathering Network" (AAAI).  PaddleOCR ships PGNet with a ResNet/FPN
backbone and heads for TCL, TBO, TDO, and TCC maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGNet(nn.Module):
    """Compact PGNet with feature pyramid and four point-gathering heads."""

    def __init__(self, classes: int = 16) -> None:
        """Initialize compact ResNet-style stem, FPN fusion, and PG heads.

        Parameters
        ----------
        classes:
            Number of character classes for the TCC head.
        """

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.res2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)
        )
        self.fuse = nn.Conv2d(96, 32, 1)
        self.tcl = nn.Conv2d(32, 1, 1)
        self.tbo = nn.Conv2d(32, 4, 1)
        self.tdo = nn.Conv2d(32, 2, 1)
        self.tcc = nn.Conv2d(32, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict PGNet center-line, border, direction, and character maps.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        torch.Tensor
            Concatenated PGNet output maps.
        """

        c1 = self.stem(image)
        c2 = F.relu(self.res2(c1))
        c3 = F.relu(self.res3(c2))
        up = F.interpolate(c3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        feat = F.relu(self.fuse(torch.cat([up, c2], dim=1)))
        return torch.cat(
            [torch.sigmoid(self.tcl(feat)), self.tbo(feat), self.tdo(feat), self.tcc(feat)], dim=1
        )


def build() -> nn.Module:
    """Build compact PGNet-ResNet50-PaddleOCR.

    Returns
    -------
    nn.Module
        Random-initialized PGNet module.
    """

    return PGNet()


def example_input() -> torch.Tensor:
    """Create a small RGB text-spotting image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 64)``.
    """

    return torch.randn(1, 3, 32, 64)


MENAGERIE_ENTRIES = [("PGNet-ResNet50-PaddleOCR", "build", "example_input", "2021", "E5")]
