"""PaddleSeg BiSeNet: bilateral real-time semantic segmentation.

Paper: Yu et al. 2018, "BiSeNet: Bilateral Segmentation Network for Real-time
Semantic Segmentation".  The compact model keeps the spatial path, context path,
attention refinement, and feature-fusion module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiSeNet(nn.Module):
    """Compact BiSeNet with spatial/context paths and feature fusion."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize bilateral branches and fusion head.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        """

        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.arm = nn.Conv2d(32, 32, 1)
        self.fuse = nn.Conv2d(48, 24, 1)
        self.attn = nn.Conv2d(24, 24, 1)
        self.head = nn.Conv2d(24, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment an image using fused spatial and context features.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        spatial = self.spatial(image)
        context = self.context(image)
        refined = context * torch.sigmoid(self.arm(F.adaptive_avg_pool2d(context, 1)))
        up = F.interpolate(refined, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        fused = F.relu(self.fuse(torch.cat([spatial, up], dim=1)))
        fused = fused * torch.sigmoid(self.attn(F.adaptive_avg_pool2d(fused, 1))) + fused
        return F.interpolate(
            self.head(fused), size=image.shape[-2:], mode="bilinear", align_corners=False
        )


def build() -> nn.Module:
    """Build compact PaddleSeg BiSeNet.

    Returns
    -------
    nn.Module
        Random-initialized BiSeNet.
    """

    return BiSeNet()


def example_input() -> torch.Tensor:
    """Create a small RGB segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("paddleseg_bisenet", "build", "example_input", "2018", "E5")]
