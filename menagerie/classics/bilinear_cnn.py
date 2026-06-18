"""Bilinear-CNN, 2015, Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.

Paper: Bilinear CNN Models for Fine-grained Visual Recognition.
Two convolutional streams are combined by orderless outer-product pooling,
then signed-square-root and L2 normalization feed a linear classifier.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BilinearCNN(nn.Module):
    """Small two-stream bilinear CNN."""

    def __init__(self, channels: int = 8, num_classes: int = 5) -> None:
        """Initialize two feature streams and the bilinear classifier.

        Parameters
        ----------
        channels:
            Feature channels in each stream.
        num_classes:
            Number of classifier outputs.
        """
        super().__init__()
        self.stream_a = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.stream_b = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(channels * channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute bilinear pooled logits.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        feat_a = self.stream_a(x)
        feat_b = self.stream_b(x)
        batch, channels_a, height, width = feat_a.shape
        channels_b = feat_b.shape[1]
        flat_a = feat_a.view(batch, channels_a, height * width)
        flat_b = feat_b.view(batch, channels_b, height * width)
        phi = torch.bmm(flat_a, flat_b.transpose(1, 2)) / float(height * width)
        signed = torch.sign(phi) * torch.sqrt(torch.abs(phi) + 1.0e-8)
        normalized = F.normalize(signed.view(batch, channels_a * channels_b), dim=1)
        return self.classifier(normalized)


def build() -> nn.Module:
    """Build a compact Bilinear-CNN.

    Returns
    -------
    nn.Module
        Random-initialized bilinear classifier.
    """
    return BilinearCNN()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)
