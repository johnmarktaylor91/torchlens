"""PaddleSeg PP-LiteSeg: real-time SPPM/UAFM segmentation.

Paper: Peng et al. 2022, "PP-LiteSeg: A Superior Real-Time Semantic
Segmentation Model".  The compact model includes Simple Pyramid Pooling (SPPM)
and Unified Attention Fusion (UAFM) in a lightweight encoder-decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPLiteSeg(nn.Module):
    """Compact PP-LiteSeg with SPPM and UAFM."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize lightweight encoder, SPPM, UAFM, and segmentation head.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        """

        super().__init__()
        self.low = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.high = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.pool_proj = nn.Conv2d(32 * 3, 32, 1)
        self.attn = nn.Conv2d(48, 16, 1)
        self.head = nn.Conv2d(16, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment an image with pyramid pooling and attention fusion.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        low = F.relu(self.low(image))
        high = F.relu(self.high(low))
        pools = [high]
        for size in (1, 2):
            pooled = F.adaptive_avg_pool2d(high, size)
            pools.append(
                F.interpolate(pooled, size=high.shape[-2:], mode="bilinear", align_corners=False)
            )
        sppm = F.relu(self.pool_proj(torch.cat(pools, dim=1)))
        up = F.interpolate(sppm, size=low.shape[-2:], mode="bilinear", align_corners=False)
        weight = torch.sigmoid(self.attn(torch.cat([low, up], dim=1)))
        fused = low * weight + up[:, :16] * (1.0 - weight)
        return F.interpolate(
            self.head(fused), size=image.shape[-2:], mode="bilinear", align_corners=False
        )


def build() -> nn.Module:
    """Build compact PaddleSeg PP-LiteSeg.

    Returns
    -------
    nn.Module
        Random-initialized PP-LiteSeg.
    """

    return PPLiteSeg()


def example_input() -> torch.Tensor:
    """Create a small segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("paddleseg_pp_liteseg", "build", "example_input", "2022", "E5")]
