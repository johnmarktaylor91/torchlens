"""DDRNet dual-resolution semantic segmenter.

Pan et al. (2022), "Deep Dual-Resolution Networks for Real-Time and Accurate
Semantic Segmentation of Road Scenes."  DDRNet maintains a high-resolution
detail branch and a deeper low-resolution semantic branch, repeatedly fusing
them bilaterally, then applies Deep Aggregation Pyramid Pooling (DAPPM).  This
compact reconstruction keeps both branches, bilateral fusion, and DAPPM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        stride:
            Spatial stride.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.net(x)


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module."""

    def __init__(self, channels: int) -> None:
        """Initialize pooling branches.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.scales = (1, 2, 4)
        self.branches = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in self.scales])
        self.process = ConvBNReLU(channels, channels)
        self.fuse = nn.Conv2d(channels * (len(self.scales) + 1), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multi-scale context.

        Parameters
        ----------
        x:
            Low-resolution semantic features.

        Returns
        -------
        torch.Tensor
            Context-enriched features.
        """

        outs = [x]
        running = x
        for scale, branch in zip(self.scales, self.branches, strict=True):
            pooled = F.adaptive_avg_pool2d(x, scale)
            pooled = F.interpolate(branch(pooled), size=x.shape[2:], mode="bilinear")
            running = self.process(running + pooled)
            outs.append(running)
        return self.fuse(torch.cat(outs, dim=1))


class CompactDDRNet(nn.Module):
    """Compact DDRNet with bilateral high/low-resolution fusion."""

    def __init__(self, classes: int = 7, width: int = 16) -> None:
        """Initialize the model.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        width:
            High-resolution branch width.
        """

        super().__init__()
        self.stem = ConvBNReLU(3, width, stride=2)
        self.high1 = ConvBNReLU(width, width)
        self.low1 = ConvBNReLU(width, width * 2, stride=2)
        self.high_to_low = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.low_to_high = nn.Conv2d(width * 2, width, 1)
        self.high2 = ConvBNReLU(width, width)
        self.low2 = ConvBNReLU(width * 2, width * 2)
        self.dappm = DAPPM(width * 2)
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment an image.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        high = self.high1(self.stem(x))
        low = self.low1(high)
        low = self.low2(low + self.high_to_low(high))
        high = self.high2(
            high + F.interpolate(self.low_to_high(low), size=high.shape[2:], mode="bilinear")
        )
        low_context = self.dappm(low)
        fused = high + F.interpolate(
            self.low_to_high(low_context), size=high.shape[2:], mode="bilinear"
        )
        return F.interpolate(self.head(fused), size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build compact DDRNet.

    Returns
    -------
    nn.Module
        Random-init DDRNet in evaluation mode.
    """

    return CompactDDRNet().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 48, 48)``.
    """

    return torch.randn(1, 3, 48, 48)


MENAGERIE_ENTRIES = [
    ("paddleseg_ddrnet", "build", "example_input", "2022", "DC"),
]
