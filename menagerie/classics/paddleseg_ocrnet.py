"""PaddleSeg OCRNet: object-contextual representations for segmentation.

Paper: Yuan et al. 2019/2020, "Object-Contextual Representations for Semantic
Segmentation".  The compact model learns coarse object regions, aggregates
class-wise object context, and augments each pixel with that context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCRNet(nn.Module):
    """Compact OCRNet with object-region context aggregation."""

    def __init__(self, classes: int = 5, dim: int = 24) -> None:
        """Initialize pixel encoder, auxiliary classifier, OCR, and head.

        Parameters
        ----------
        classes:
            Number of object classes.
        dim:
            Pixel feature width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.ReLU(), nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU()
        )
        self.aux = nn.Conv2d(dim, classes, 1)
        self.object_proj = nn.Linear(dim, dim)
        self.pixel_proj = nn.Conv2d(dim * 2, dim, 1)
        self.head = nn.Conv2d(dim, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict segmentation logits with object-contextual features.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        feat = self.encoder(image)
        coarse = torch.softmax(self.aux(feat), dim=1)
        flat_feat = feat.flatten(2).transpose(1, 2)
        flat_prob = coarse.flatten(2)
        objects = torch.bmm(flat_prob, flat_feat) / flat_prob.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        context = (
            torch.bmm(flat_prob.transpose(1, 2), self.object_proj(objects))
            .transpose(1, 2)
            .view_as(feat)
        )
        return self.head(F.relu(self.pixel_proj(torch.cat([feat, context], dim=1))))


def build() -> nn.Module:
    """Build compact PaddleSeg OCRNet.

    Returns
    -------
    nn.Module
        Random-initialized OCRNet.
    """

    return OCRNet()


def example_input() -> torch.Tensor:
    """Create a small segmentation image.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 24, 24)``.
    """

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [("paddleseg_ocrnet", "build", "example_input", "2019", "E5")]
