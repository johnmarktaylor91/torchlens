"""Dynamic Convolution: attention over convolution kernels.

Paper: Dynamic Convolution: Attention over Convolution Kernels.
Chen, Dai, Liu, Chen, Yuan, and Liu, CVPR 2020.

Dynamic convolution keeps several parallel kernels and uses an input-dependent
attention vector to aggregate them into the convolution applied to each sample.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DynamicConv2d(nn.Module):
    """Input-conditioned aggregation over a bank of convolution kernels."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 4) -> None:
        """Initialize kernel bank and routing attention.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        num_kernels:
            Number of parallel kernels in the bank.
        """
        super().__init__()
        self.padding = 1
        self.weight = nn.Parameter(torch.randn(num_kernels, out_channels, in_channels, 3, 3) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels))
        hidden = max(4, in_channels // 4)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_kernels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply per-sample dynamic convolution.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Dynamically convolved feature map.
        """
        routing = torch.softmax(self.router(x), dim=1)
        outputs = []
        for idx in range(x.shape[0]):
            weight = torch.sum(routing[idx, :, None, None, None, None] * self.weight, dim=0)
            bias = torch.sum(routing[idx, :, None] * self.bias, dim=0)
            outputs.append(F.conv2d(x[idx : idx + 1], weight, bias=bias, padding=self.padding))
        return torch.cat(outputs, dim=0)


class DynamicConvDemoNet(nn.Module):
    """Compact lightweight CNN using dynamic convolution."""

    def __init__(self, channels: int = 12, num_classes: int = 5) -> None:
        """Initialize the demonstration classifier.

        Parameters
        ----------
        channels:
            Width of the convolutional trunk.
        num_classes:
            Number of output logits.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.dynamic = DynamicConv2d(channels, channels)
        self.norm = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits through a dynamic convolution block.

        Parameters
        ----------
        x:
            RGB image tensor with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = self.stem(x)
        x = torch.relu(self.norm(self.dynamic(x)))
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.classifier(x)


def build() -> nn.Module:
    """Build a compact dynamic-convolution demonstration network.

    Returns
    -------
    nn.Module
        Random-initialized dynamic-convolution demo network.
    """
    return DynamicConvDemoNet()


def example_input() -> Tensor:
    """Return a small traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Dynamic Convolution (attention over convolution kernels)",
        "build",
        "example_input",
        "2020",
        "DC",
    )
]
