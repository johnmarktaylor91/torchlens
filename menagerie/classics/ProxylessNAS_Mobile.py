"""ProxylessNAS-Mobile compact searched mobile network.

Paper: ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.

Cai et al. (ICLR 2019) searched directly over mobile inverted bottleneck blocks
using binarized path gates and latency-aware objectives.  The exported Mobile
architecture is a MobileNetV2-like sequence of MBConv choices with depthwise
separable convolutions, squeeze/linear bottlenecks, and residual shortcuts.
This compact random-init module keeps the chosen-path MBConv structure.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MBConv(nn.Module):
    """Mobile inverted bottleneck convolution block."""

    def __init__(self, in_ch: int, out_ch: int, expansion: int, stride: int, kernel: int) -> None:
        """Initialize an MBConv candidate chosen by ProxylessNAS.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        expansion:
            Expansion ratio.
        stride:
            Depthwise stride.
        kernel:
            Depthwise kernel size.
        """

        super().__init__()
        mid = in_ch * expansion
        pad = kernel // 2
        self.use_residual = stride == 1 and in_ch == out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=False),
            nn.Conv2d(mid, mid, kernel, stride=stride, padding=pad, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=False),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply MBConv and optional inverted residual shortcut.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """

        y = self.net(x)
        return x + y if self.use_residual else y


class ProxylessNASMobile(nn.Module):
    """Compact ProxylessNAS-Mobile classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize searched mobile stages.

        Parameters
        ----------
        classes:
            Number of classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=False),
        )
        self.blocks = nn.Sequential(
            MBConv(24, 24, 3, 1, 3),
            MBConv(24, 32, 3, 2, 5),
            MBConv(32, 32, 6, 1, 3),
            MBConv(32, 48, 6, 2, 7),
            MBConv(48, 48, 3, 1, 5),
            MBConv(48, 80, 6, 2, 3),
        )
        self.head = nn.Sequential(
            nn.Conv2d(80, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, classes),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        return self.head(self.blocks(self.stem(image)))


def build() -> nn.Module:
    """Build compact random-init ProxylessNAS-Mobile.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return ProxylessNASMobile().eval()


def example_input() -> Tensor:
    """Return a small RGB input.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ProxylessNAS-Mobile", "build", "example_input", "2019", "CV")]
