"""PaddleDetection GFL: dense detector with quality and distributional boxes.

Li et al. (NeurIPS 2020), "Generalized Focal Loss: Learning Qualified and
Distributed Bounding Boxes for Dense Object Detection".  GFL jointly represents
classification quality and class confidence, and predicts box side locations as
discrete distributions optimized by Distribution Focal Loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaddleDetGFL(nn.Module):
    """Compact GFL detector."""

    def __init__(self, classes: int = 5, bins: int = 8, channels: int = 32) -> None:
        """Initialize GFL.

        Parameters
        ----------
        classes:
            Number of object classes.
        bins:
            Number of distance bins per box side.
        channels:
            Feature channel count.
        """

        super().__init__()
        self.bins = bins
        self.register_buffer("project", torch.arange(bins, dtype=torch.float32))
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.quality_cls = nn.Conv2d(channels, classes, 3, padding=1)
        self.distribution = nn.Conv2d(channels, 4 * bins, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict joint quality-class scores and distributed box distances.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Quality-aware class scores, side distributions, and expected boxes.
        """

        feat = self.tower(self.backbone(x))
        dist = self.distribution(feat)
        batch, _, height, width = dist.shape
        dist = dist.view(batch, 4, self.bins, height, width).softmax(dim=2)
        expected = (dist * self.project.view(1, 1, self.bins, 1, 1)).sum(dim=2)
        return {
            "quality_cls": torch.sigmoid(self.quality_cls(feat)),
            "box_distribution": dist,
            "expected_ltrb": expected,
        }


def build() -> nn.Module:
    """Build the compact PaddleDetection GFL model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetGFL().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("paddledet_gfl", "build", "example_input", "2020", "DC")]
