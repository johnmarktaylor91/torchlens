"""PaddleSeg U2-Net salient-object segmenter.

Qin et al. (2020), "U2-Net: Going Deeper with Nested U-Structure for Salient
Object Detection."  U2-Net is an outer encoder-decoder whose stages are
ReSidual U-blocks (RSU): each RSU is itself a small U-Net with a residual skip
from block input to block output.  This compact reconstruction keeps the nested
RSU encoder/decoder, side-output saliency heads, and fused saliency head, with
reduced widths and depth for a fast random-init TorchLens render.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        dilation:
            Dilation factor for the spatial convolution.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x:
            Input image feature map.

        Returns
        -------
        torch.Tensor
            Transformed feature map.
        """

        return self.net(x)


class MiniRSU(nn.Module):
    """Compact residual U-block used by U2-Net stages."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        """Initialize a two-level residual U-block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        mid_channels:
            Internal channel count.
        out_channels:
            Output channel count.
        """

        super().__init__()
        self.in_conv = ConvBNReLU(in_channels, out_channels)
        self.enc1 = ConvBNReLU(out_channels, mid_channels)
        self.enc2 = ConvBNReLU(mid_channels, mid_channels)
        self.bridge = ConvBNReLU(mid_channels, mid_channels, dilation=2)
        self.dec2 = ConvBNReLU(mid_channels * 2, mid_channels)
        self.dec1 = ConvBNReLU(mid_channels * 2, out_channels)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run nested U-shaped encoding and decoding.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual U-block output.
        """

        residual = self.in_conv(x)
        e1 = self.enc1(residual)
        e2 = self.enc2(self.pool(e1))
        b = self.bridge(self.pool(e2))
        d2 = self.dec2(torch.cat([F.interpolate(b, size=e2.shape[2:], mode="bilinear"), e2], dim=1))
        d1 = self.dec1(
            torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear"), e1], dim=1)
        )
        return d1 + residual


class CompactU2Net(nn.Module):
    """Small U2-Net with nested RSU blocks and side saliency outputs."""

    def __init__(self, channels: int = 12) -> None:
        """Initialize the compact U2-Net.

        Parameters
        ----------
        channels:
            Base feature width.
        """

        super().__init__()
        self.stage1 = MiniRSU(3, channels, channels * 2)
        self.stage2 = MiniRSU(channels * 2, channels, channels * 4)
        self.stage3 = MiniRSU(channels * 4, channels * 2, channels * 4)
        self.stage2d = MiniRSU(channels * 8, channels, channels * 2)
        self.stage1d = MiniRSU(channels * 4, channels, channels * 2)
        self.side1 = nn.Conv2d(channels * 2, 1, 3, padding=1)
        self.side2 = nn.Conv2d(channels * 2, 1, 3, padding=1)
        self.side3 = nn.Conv2d(channels * 4, 1, 3, padding=1)
        self.fuse = nn.Conv2d(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a fused saliency map from an image.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Saliency probabilities at input resolution.
        """

        e1 = self.stage1(x)
        e2 = self.stage2(F.max_pool2d(e1, 2, ceil_mode=True))
        e3 = self.stage3(F.max_pool2d(e2, 2, ceil_mode=True))
        d2 = self.stage2d(
            torch.cat([F.interpolate(e3, size=e2.shape[2:], mode="bilinear"), e2], dim=1)
        )
        d1 = self.stage1d(
            torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear"), e1], dim=1)
        )
        s1 = self.side1(d1)
        s2 = F.interpolate(self.side2(d2), size=x.shape[2:], mode="bilinear")
        s3 = F.interpolate(self.side3(e3), size=x.shape[2:], mode="bilinear")
        return torch.sigmoid(self.fuse(torch.cat([s1, s2, s3], dim=1)))


def build() -> nn.Module:
    """Build the compact PaddleSeg U2-Net model.

    Returns
    -------
    nn.Module
        Random-init U2-Net in evaluation mode.
    """

    return CompactU2Net().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image for tracing.

    Returns
    -------
    torch.Tensor
        Image batch of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("ppseg_u2net", "build", "example_input", "2020", "DC"),
]
