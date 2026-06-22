"""DetNet: detection-oriented high-resolution dilated CNN backbone.

Paper: DetNet: A Backbone network for Object Detection.
Li, Peng, Yu, Zhang, Deng, and Sun, ECCV 2018.

DetNet keeps deeper backbone stages at stride 16 instead of continuing to stride
32, then uses bottleneck blocks with dilated 3x3 convolutions and projection
shortcuts to enlarge receptive field while preserving localization resolution.
This compact classifier head exposes the same backbone behavior for tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    """Convolution, batch normalization, and ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        kernel_size:
            Spatial convolution kernel size.
        stride:
            Convolution stride.
        padding:
            Zero-padding size.
        dilation:
            Convolution dilation rate.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Activated output feature map.
        """

        return self.net(x)


class DetBottleneck(nn.Module):
    """DetNet bottleneck with optional projection and dilated spatial convolution."""

    def __init__(self, channels: int, dilation: int = 1, projection: bool = False) -> None:
        """Initialize the DetNet bottleneck.

        Parameters
        ----------
        channels:
            Input and output channel count.
        dilation:
            Dilation rate for the 3x3 convolution.
        projection:
            Whether to use the 1x1 projection shortcut from DetNet stage starts.
        """

        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvBnRelu(channels, mid_channels, 1, padding=0)
        self.conv2 = ConvBnRelu(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
            if projection
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a DetNet bottleneck.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual bottleneck output.
        """

        residual = self.shortcut(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(x + residual)


class CompactDetNet(nn.Module):
    """Small DetNet-style backbone with stride-16 dilated deep stages."""

    def __init__(self, num_classes: int = 10, channels: int = 32) -> None:
        """Initialize the compact DetNet.

        Parameters
        ----------
        num_classes:
            Number of classifier outputs.
        channels:
            Internal feature width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(3, channels // 2, stride=2),
            ConvBnRelu(channels // 2, channels, stride=2),
            ConvBnRelu(channels, channels, stride=2),
            ConvBnRelu(channels, channels, stride=2),
        )
        self.det_stage_5 = nn.Sequential(
            DetBottleneck(channels, dilation=2, projection=True),
            DetBottleneck(channels, dilation=2),
        )
        self.det_stage_6 = nn.Sequential(
            DetBottleneck(channels, dilation=4, projection=True),
            DetBottleneck(channels, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Image tensor with shape ``(B, 3, 64, 64)``.

        Returns
        -------
        torch.Tensor
            Class logits from the stride-16 dilated detection backbone.
        """

        x = self.stem(x)
        x = self.det_stage_5(x)
        x = self.det_stage_6(x)
        return self.head(self.pool(x).flatten(1))


def build() -> nn.Module:
    """Build the compact DetNet classic.

    Returns
    -------
    nn.Module
        Evaluation-mode compact DetNet model.
    """

    return CompactDetNet().eval()


def example_input() -> torch.Tensor:
    """Create the canonical example image batch.

    Returns
    -------
    torch.Tensor
        Random image tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "DetNet (dilated high-resolution detection backbone)",
        "build",
        "example_input",
        "2018",
        "E5",
    ),
]
