"""PaddleSeg UPerNet semantic-segmentation head.

Xiao et al. (2018), "Unified Perceptual Parsing for Scene Understanding."
UPerNet combines a backbone feature hierarchy with a PSPNet-style Pyramid
Pooling Module on the deepest feature map, a top-down Feature Pyramid Network,
and a final segmentation head.  This compact reconstruction keeps those three
signature pieces with a tiny CNN backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        stride:
            Spatial convolution stride.
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


class PyramidPooling(nn.Module):
    """PSPNet-style pyramid pooling module used by UPerNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize pooled projection branches.

        Parameters
        ----------
        in_channels:
            Deepest feature channel count.
        out_channels:
            Output channel count for each pooled branch and fused result.
        """

        super().__init__()
        self.scales = (1, 2, 3)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(in_channels, out_channels, 1))
                for scale in self.scales
            ]
        )
        self.fuse = nn.Conv2d(in_channels + len(self.scales) * out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool at multiple bin sizes and concatenate with the input.

        Parameters
        ----------
        x:
            Deep feature map.

        Returns
        -------
        torch.Tensor
            Pyramid-pooled feature map.
        """

        pooled = [x]
        for branch in self.branches:
            pooled.append(F.interpolate(branch(x), size=x.shape[2:], mode="bilinear"))
        return self.fuse(torch.cat(pooled, dim=1))


class CompactUPerNet(nn.Module):
    """Compact UPerNet with PPM plus top-down FPN fusion."""

    def __init__(self, classes: int = 7, width: int = 16) -> None:
        """Initialize UPerNet.

        Parameters
        ----------
        classes:
            Number of semantic classes.
        width:
            Base feature width.
        """

        super().__init__()
        self.c1 = ConvBNAct(3, width, stride=2)
        self.c2 = ConvBNAct(width, width * 2, stride=2)
        self.c3 = ConvBNAct(width * 2, width * 4, stride=2)
        self.ppm = PyramidPooling(width * 4, width)
        self.lat3 = nn.Conv2d(width, width, 1)
        self.lat2 = nn.Conv2d(width * 2, width, 1)
        self.lat1 = nn.Conv2d(width, width, 1)
        self.smooth3 = ConvBNAct(width, width)
        self.smooth2 = ConvBNAct(width, width)
        self.head = nn.Conv2d(width * 3, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Semantic logits at input resolution.
        """

        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        p3 = self.smooth3(self.lat3(self.ppm(c3)))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear"))
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear")
        out = self.head(
            torch.cat(
                [
                    p1,
                    F.interpolate(p2, size=p1.shape[2:], mode="bilinear"),
                    F.interpolate(p3, size=p1.shape[2:], mode="bilinear"),
                ],
                dim=1,
            )
        )
        return F.interpolate(out, size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build the compact PaddleSeg UPerNet.

    Returns
    -------
    nn.Module
        Random-init segmentation model in evaluation mode.
    """

    return CompactUPerNet().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddleseg_upernet", "build", "example_input", "2018", "DC"),
]
