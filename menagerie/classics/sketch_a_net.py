"""Sketch-a-Net, 2015, Qian Yu et al.

Paper: Sketch-a-Net: A Deep Neural Network that Beats Humans.
An AlexNet-like sketch recognizer uses a large 15x15 stride-3 first filter,
no LRN, and convolutional fully connected layers for sketch categories.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SketchANet(nn.Module):
    """Sketch-tuned AlexNet variant."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize Sketch-a-Net layers.

        Parameters
        ----------
        num_classes:
            Number of sketch classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=15, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Classify a free-hand sketch image.

        Parameters
        ----------
        x:
            Grayscale sketch tensor ``(B, 1, 225, 225)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        y = self.classifier(self.features(x))
        return y.flatten(1)


def build() -> nn.Module:
    """Build a compact Sketch-a-Net.

    Returns
    -------
    nn.Module
        Random-initialized SketchANet.
    """
    return SketchANet()


def example_input() -> Tensor:
    """Return a traceable sketch batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 225, 225)``.
    """
    return torch.randn(1, 1, 225, 225)
