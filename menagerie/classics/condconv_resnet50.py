"""CondConv-ResNet: conditionally parameterized convolutions in a ResNet block.

Yang et al., NeurIPS 2019, arXiv:1904.04971.  CondConv replaces a shared
convolution kernel with an example-dependent weighted mixture of expert kernels.
This compact random-init reconstruction places CondConv inside a small
ResNet-style bottleneck classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondConv2d(nn.Module):
    """Per-example mixture of convolution expert kernels."""

    def __init__(self, channels: int, experts: int = 4) -> None:
        """Initialize CondConv experts and routing.

        Parameters
        ----------
        channels:
            Input and output channel count.
        experts:
            Number of expert kernels.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.randn(experts, channels, channels, 3, 3) * 0.02)
        self.bias = nn.Parameter(torch.zeros(experts, channels))
        self.router = nn.Linear(channels, experts)
        self.padding = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply example-conditioned convolution.

        Parameters
        ----------
        x:
            Input image features.

        Returns
        -------
        torch.Tensor
            Convolved features.
        """

        gates = torch.sigmoid(self.router(x.mean(dim=(2, 3))))
        outs = []
        for batch_idx in range(x.shape[0]):
            weight = torch.sum(gates[batch_idx, :, None, None, None, None] * self.weight, dim=0)
            bias = torch.sum(gates[batch_idx, :, None] * self.bias, dim=0)
            outs.append(F.conv2d(x[batch_idx : batch_idx + 1], weight, bias, padding=self.padding))
        return torch.cat(outs, dim=0)


class CondConvBottleneck(nn.Module):
    """Compact ResNet bottleneck containing CondConv."""

    def __init__(self, channels: int) -> None:
        """Initialize bottleneck layers.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.reduce = nn.Conv2d(channels, channels, 1)
        self.cond = CondConv2d(channels)
        self.expand = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual bottleneck.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        y = F.relu(self.reduce(x))
        y = F.relu(self.cond(y))
        y = self.norm(self.expand(y))
        return F.relu(x + y)


class CompactCondConvResNet(nn.Module):
    """Small CondConv-ResNet classifier."""

    def __init__(self) -> None:
        """Initialize the compact classifier."""

        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.block1 = CondConvBottleneck(32)
        self.block2 = CondConvBottleneck(32)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = F.relu(self.stem(x))
        x = self.block2(self.block1(x))
        return self.head(x.mean(dim=(2, 3)))


def build() -> nn.Module:
    """Build compact CondConv-ResNet.

    Returns
    -------
    nn.Module
        Random-init CondConv-ResNet reconstruction.
    """

    return CompactCondConvResNet()


def example_input() -> torch.Tensor:
    """Create an image input.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(2, 3, 32, 32)


MENAGERIE_ENTRIES = [("CondConv-ResNet50", "build", "example_input", "2019", "CV")]
