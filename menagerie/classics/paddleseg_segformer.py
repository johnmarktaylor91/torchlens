"""PaddleSeg SegFormer: hierarchical transformer with all-MLP decoder.

Paper: Xie et al. 2021, "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers".  The compact model keeps overlapping patch
embeddings, hierarchical transformer stages, and the lightweight MLP decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormer(nn.Module):
    """Compact SegFormer segmentation model."""

    def __init__(self, classes: int = 5, dim: int = 32) -> None:
        """Initialize patch stages, transformer block, and MLP decoder.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.stage1 = nn.Conv2d(3, dim, 7, stride=4, padding=3)
        self.stage2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.decode1 = nn.Conv2d(dim, dim, 1)
        self.decode2 = nn.Conv2d(dim, dim, 1)
        self.head = nn.Conv2d(dim * 2, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment an image using hierarchical transformer features.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        f1 = F.relu(self.stage1(image))
        f2 = F.relu(self.stage2(f1))
        tokens = f2.flatten(2).transpose(1, 2)
        encoded = self.transformer(tokens).transpose(1, 2).view_as(f2)
        d2 = F.interpolate(
            self.decode2(encoded), size=f1.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.head(torch.cat([self.decode1(f1), d2], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build() -> nn.Module:
    """Build compact PaddleSeg SegFormer.

    Returns
    -------
    nn.Module
        Random-initialized SegFormer.
    """

    return SegFormer()


def example_input() -> torch.Tensor:
    """Create a small segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("paddleseg_segformer", "build", "example_input", "2021", "E5")]
