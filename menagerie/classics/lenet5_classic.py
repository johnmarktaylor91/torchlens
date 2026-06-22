"""LeNet-5 classic convolutional digit recognizer.

LeCun et al., 1998.
Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

LeNet-5 uses C1/S2/C3/S4/C5/F6 layers: learned convolution, average
subsampling, a partially connected C3 bank, more subsampling, and a final
classifier.  The compact reconstruction keeps the partial C3 connection table
instead of replacing the model with a generic CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialC3(nn.Module):
    """LeNet-5 C3 layer with a fixed partial connection table."""

    def __init__(self) -> None:
        """Initialize 16 partially connected 5x5 filters."""

        super().__init__()
        table = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ]
        self.register_buffer("mask", torch.zeros(16, 6, 1, 1))
        for out_idx, inputs in enumerate(table):
            self.mask[out_idx, inputs, 0, 0] = 1.0
        self.weight = nn.Parameter(torch.randn(16, 6, 5, 5) * 0.05)
        self.bias = nn.Parameter(torch.zeros(16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply partially connected C3 convolution.

        Parameters
        ----------
        x:
            S2 feature maps with shape ``(batch, 6, 14, 14)``.

        Returns
        -------
        torch.Tensor
            C3 activations.
        """

        return F.conv2d(x, self.weight * self.mask, self.bias)


class LeNet5Classic(nn.Module):
    """Classic LeNet-5 with average subsampling and partial C3."""

    def __init__(self) -> None:
        """Initialize LeNet-5 layers."""

        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2_scale = nn.Parameter(torch.ones(6))
        self.s2_bias = nn.Parameter(torch.zeros(6))
        self.c3 = PartialC3()
        self.s4_scale = nn.Parameter(torch.ones(16))
        self.s4_bias = nn.Parameter(torch.zeros(16))
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def _subsample(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Run learned average subsampling.

        Parameters
        ----------
        x:
            Input feature maps.
        scale:
            Per-channel learned scale.
        bias:
            Per-channel learned bias.

        Returns
        -------
        torch.Tensor
            Subsampled activations.
        """

        y = F.avg_pool2d(x, kernel_size=2, stride=2)
        return torch.tanh(y * scale.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify 32x32 grayscale images.

        Parameters
        ----------
        x:
            Image tensor with shape ``(batch, 1, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Digit logits.
        """

        x = torch.tanh(self.c1(x))
        x = self._subsample(x, self.s2_scale, self.s2_bias)
        x = torch.tanh(self.c3(x))
        x = self._subsample(x, self.s4_scale, self.s4_bias)
        x = torch.tanh(self.c5(x)).flatten(1)
        x = torch.tanh(self.f6(x))
        return self.out(x)


def build() -> nn.Module:
    """Build a classic LeNet-5 reconstruction.

    Returns
    -------
    nn.Module
        Random-init LeNet-5.
    """

    return LeNet5Classic()


def example_input() -> torch.Tensor:
    """Create a LeNet-5 image input.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 1, 32, 32)``.
    """

    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("LeNet5-classic", "build", "example_input", "1998", "CC"),
]
