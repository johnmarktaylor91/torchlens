"""DeconvNet, 2015, Hyeonwoo Noh, Seunghoon Hong, Bohyung Han.

Paper: Learning Deconvolution Network for Semantic Segmentation.
A VGG-style encoder records max-pool switches and a mirrored decoder restores
resolution with max-unpooling followed by convolutional refinement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeconvNetNoh(nn.Module):
    """Small switch-index unpooling encoder-decoder segmentation network."""

    def __init__(self, num_classes: int = 4) -> None:
        """Initialize encoder, unpooling decoder, and classifier.

        Parameters
        ----------
        num_classes:
            Number of segmentation classes.
        """
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.classifier = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Segment an image using stored max-pooling switches.

        Parameters
        ----------
        x:
            RGB tensor with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Segmentation logits.
        """
        x1 = self.enc1(x)
        p1, i1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2, i2 = self.pool(x2)
        x3 = self.enc3(p2)
        p3, i3 = self.pool(x3)
        y = self.unpool(p3, i3, output_size=x3.shape)
        y = self.dec3(y)
        y = self.unpool(y, i2, output_size=x2.shape)
        y = self.dec2(y)
        y = self.unpool(y, i1, output_size=x1.shape)
        y = self.dec1(y)
        return self.classifier(y)


def build() -> nn.Module:
    """Build a compact DeconvNet.

    Returns
    -------
    nn.Module
        Random-initialized DeconvNet.
    """
    return DeconvNetNoh()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)
