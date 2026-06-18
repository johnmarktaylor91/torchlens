"""LeNet-4 / pre-LeNet-5 CNN, 1995, Yann LeCun et al.

Paper: Gradient-based learning applied to document recognition.
Intermediate convolution/subsampling digit recognizer with tanh feature maps,
learned average-pooling gains, and a compact classifier head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScaledAvgPool2d(nn.Module):
    """Average-pooling layer with learned per-channel scale and bias."""

    def __init__(self, channels: int, kernel_size: int = 2) -> None:
        """Initialize the learned subsampling layer.

        Parameters
        ----------
        channels:
            Number of feature-map channels.
        kernel_size:
            Spatial averaging window and stride.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        """Apply learned scaled average subsampling.

        Parameters
        ----------
        x:
            Input feature maps of shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Subsampled feature maps.
        """
        pooled = F.avg_pool2d(x, self.kernel_size, self.kernel_size)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return torch.tanh(pooled * weight + bias)


class LeNet4(nn.Module):
    """Compact LeNet-4 style digit classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize LeNet-4 layers.

        Parameters
        ----------
        num_classes:
            Number of output digit classes.
        """
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, kernel_size=5)
        self.s2 = ScaledAvgPool2d(4)
        self.c3 = nn.Conv2d(4, 16, kernel_size=5)
        self.s4 = ScaledAvgPool2d(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a 32 by 32 grayscale image.

        Parameters
        ----------
        x:
            Batch of images with shape ``(B, 1, 32, 32)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = torch.tanh(self.c1(x))
        x = self.s2(x)
        x = torch.tanh(self.c3(x))
        x = self.s4(x)
        x = torch.tanh(self.c5(x))
        return self.classifier(torch.flatten(x, 1))


def build() -> nn.Module:
    """Build a small random-initialized LeNet-4 module.

    Returns
    -------
    nn.Module
        A LeNet-4 classifier.
    """
    return LeNet4()


def example_input() -> Tensor:
    """Return a traceable example image batch.

    Returns
    -------
    Tensor
        Float image tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
