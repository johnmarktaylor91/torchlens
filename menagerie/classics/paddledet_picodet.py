"""PP-PicoDet compact mobile object detector.

Yu et al., 2021, "PP-PicoDet: A Better Real-Time Object Detector on Mobile
Devices".  PicoDet uses an ESNet-style lightweight backbone, CSP-PAN neck,
anchor-free detection, and a GFL/quality focal style head.  This compact
random-init reconstruction keeps those inference components at tiny width.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthwiseSeparable(nn.Module):
    """Depthwise-pointwise convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the depthwise separable block.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        stride:
            Depthwise convolution stride.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """
        return self.net(x)


class PicoDet(nn.Module):
    """Compact PicoDet with ESNet/CSP-PAN/head."""

    def __init__(self, width: int = 32, classes: int = 20, bins: int = 8) -> None:
        """Initialize PicoDet.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of object classes.
        bins:
            Distribution-regression bins per box side.
        """
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.Hardswish())
        self.c3 = DepthwiseSeparable(width, width, 2)
        self.c4 = DepthwiseSeparable(width, width, 2)
        self.c5 = DepthwiseSeparable(width, width, 2)
        self.lateral = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(3)])
        self.fuse = DepthwiseSeparable(width, width)
        self.cls_head = nn.Conv2d(width, classes, 1)
        self.box_head = nn.Conv2d(width, 4 * bins, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Run anchor-free classification and distribution box heads.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Flattened class logits and box distributions.
        """
        x = self.stem(image)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        feat = self.fuse(p3)
        cls = self.cls_head(feat).flatten(2).transpose(1, 2)
        box = self.box_head(feat).flatten(2).transpose(1, 2)
        return cls, box


def build() -> nn.Module:
    """Build a compact PicoDet detector.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """
    return PicoDet().eval()


def example_input() -> Tensor:
    """Return a small mobile-detector image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_picodet", "build", "example_input", "2021", "DC"),
]
