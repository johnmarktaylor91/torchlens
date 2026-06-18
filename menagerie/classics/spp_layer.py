"""SPP-net spatial pyramid pooling, 2014, Kaiming He et al.

Paper: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.
Multi-level adaptive max pooling converts variable-size convolutional features
into a fixed-length vector for a fully connected classifier.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("SPP-net Spatial Pyramid Pooling Layer", "build", "example_input", "2014", "DC")
]


class SpatialPyramidPoolingLayer(nn.Module):
    """Small SPP-net head over convolutional feature maps."""

    def __init__(self, channels: int = 256, num_classes: int = 10) -> None:
        """Initialize fixed-level SPP classifier.

        Parameters
        ----------
        channels
            Input feature channel count.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.classifier = nn.Linear(channels * (1 + 4 + 16), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Pool feature maps at 1x1, 2x2, and 4x4 spatial levels.

        Parameters
        ----------
        x
            Feature tensor with shape ``(B, 256, 7, 7)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        pooled = [F.adaptive_max_pool2d(x, (level, level)).flatten(1) for level in (1, 2, 4)]
        return self.classifier(torch.cat(pooled, dim=1))


def build() -> nn.Module:
    """Build a compact SPP-net pooling head.

    Returns
    -------
    nn.Module
        Random-initialized SPP classifier.
    """
    return SpatialPyramidPoolingLayer()


def example_input() -> Tensor:
    """Return a traceable convolutional feature map.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 256, 7, 7)``.
    """
    return torch.randn(1, 256, 7, 7)
