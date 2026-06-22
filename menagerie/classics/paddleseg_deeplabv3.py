"""PaddleSeg DeepLabV3: atrous spatial pyramid semantic segmentation.

Paper: Chen et al. 2017, "Rethinking Atrous Convolution for Semantic Image
Segmentation".  PaddleSeg exposes DeepLabV3 with atrous backbone features and
ASPP multi-rate context aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3(nn.Module):
    """Compact DeepLabV3 with ASPP branches."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize backbone, ASPP branches, and segmentation head.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.aspp = nn.ModuleList([nn.Conv2d(32, 16, 3, padding=r, dilation=r) for r in (1, 2, 3)])
        self.image_pool = nn.Conv2d(32, 16, 1)
        self.head = nn.Conv2d(64, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment an image using atrous pyramid context.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Segmentation logits at input resolution.
        """

        feat = self.backbone(image)
        pooled = F.interpolate(
            self.image_pool(F.adaptive_avg_pool2d(feat, 1)),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        branches = [F.relu(branch(feat)) for branch in self.aspp] + [F.relu(pooled)]
        logits = self.head(torch.cat(branches, dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build() -> nn.Module:
    """Build compact PaddleSeg DeepLabV3.

    Returns
    -------
    nn.Module
        Random-initialized DeepLabV3.
    """

    return DeepLabV3()


def example_input() -> torch.Tensor:
    """Create a small segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("paddleseg_deeplabv3", "build", "example_input", "2017", "E5")]
