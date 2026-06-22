"""TTFNet compact anchor-free detector.

Paper: "Training-Time-Friendly Network for Real-Time Object Detection"
(Liu et al., AAAI 2020).

TTFNet is represented by its distinctive light, single-stage, anchor-free design:
a deconvolutional upsampling neck feeds exactly two dense heads, a heatmap head
for object centers and a box-size head for width/height regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TTFNetCompact(nn.Module):
    """Compact TTFNet with center heatmap and size heads."""

    def __init__(self, channels: int = 32, num_classes: int = 5) -> None:
        """Initialize TTFNet.

        Parameters
        ----------
        channels:
            Internal feature width.
        num_classes:
            Number of center-heatmap classes.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, num_classes, 1),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Conv2d(channels, 4, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict dense center heatmaps and box sizes.

        Parameters
        ----------
        x:
            Input image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Center heatmap logits and positive box-size distances.
        """

        feat = self.upsample(self.backbone(x))
        return self.heatmap_head(feat), torch.relu(self.wh_head(feat))


def build_paddledet_ttfnet() -> nn.Module:
    """Build compact random-init TTFNet.

    Returns
    -------
    nn.Module
        TTFNet compact model.
    """

    return TTFNetCompact().eval()


def example_input() -> torch.Tensor:
    """Create a small RGB image.

    Returns
    -------
    torch.Tensor
        Input tensor.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_ttfnet", "build_paddledet_ttfnet", "example_input", "2020", "DC"),
]
