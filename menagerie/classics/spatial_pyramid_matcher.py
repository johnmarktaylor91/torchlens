"""Spatial Pyramid Matching Network, 2006, Svetlana Lazebnik et al.

Paper: Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories.
Local descriptors are softly assigned to a codebook, pooled over progressively
finer spatial bins, concatenated, and classified.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [("Spatial Pyramid Matching Network", "build", "example_input", "2006", "DC")]


class SpatialPyramidMatchingNetwork(nn.Module):
    """Small soft-codebook spatial pyramid classifier."""

    def __init__(self, channels: int = 8, codewords: int = 6, num_classes: int = 5) -> None:
        """Initialize descriptor extractor, codebook, and classifier.

        Parameters
        ----------
        channels
            Descriptor channel count.
        codewords
            Number of visual codewords.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.descriptor = nn.Sequential(
            nn.Conv2d(3, channels, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.codebook = nn.Parameter(torch.randn(codewords, channels))
        self.classifier = nn.Linear(codewords * (1 + 4 + 16), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image with spatial-pyramid soft histograms.

        Parameters
        ----------
        x
            RGB image tensor with shape ``(B, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        desc = self.descriptor(x)
        batch, channels, height, width = desc.shape
        flat = desc.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        dist = torch.cdist(flat, self.codebook.unsqueeze(0).expand(batch, -1, -1))
        assign = torch.softmax(-dist, dim=-1).transpose(1, 2).reshape(batch, -1, height, width)
        pooled = [F.adaptive_avg_pool2d(assign, (level, level)).flatten(1) for level in (1, 2, 4)]
        return self.classifier(torch.cat(pooled, dim=1))


def build() -> nn.Module:
    """Build a compact spatial-pyramid matcher.

    Returns
    -------
    nn.Module
        Random-initialized spatial-pyramid classifier.
    """
    return SpatialPyramidMatchingNetwork()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 224, 224)``.
    """
    return torch.randn(1, 3, 224, 224)
