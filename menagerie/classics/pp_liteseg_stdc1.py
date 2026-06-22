"""PP-LiteSeg with an STDC-style compact backbone.

Peng et al. (2022), "PP-LiteSeg: A Superior Real-Time Semantic Segmentation
Model."  PP-LiteSeg uses a lightweight encoder, Simple Pyramid Pooling Module
(SPPM), Flexible and Lightweight Decoder (FLD), and Unified Attention Fusion
Modules (UAFM).  This compact version keeps the STDC-style multi-branch block,
SPPM, and two UAFM decoder fusions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
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
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.net(x)


class STDCBlock(nn.Module):
    """Short-Term Dense Concatenate block used by STDC backbones."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the STDC block.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels after concatenating branches.
        stride:
            First-branch stride.
        """

        super().__init__()
        split = out_channels // 4
        self.branch1 = ConvBNReLU(in_channels, split, stride=stride)
        self.branch2 = ConvBNReLU(split, split)
        self.branch3 = ConvBNReLU(split, split)
        self.branch4 = ConvBNReLU(split, split)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dense concatenated STDC branches.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            STDC feature map.
        """

        b1 = self.branch1(x)
        b2 = self.branch2(b1)
        b3 = self.branch3(b2)
        b4 = self.branch4(b3)
        return torch.cat([b1, b2, b3, b4], dim=1) + self.skip(x)


class SPPM(nn.Module):
    """Simple Pyramid Pooling Module from PP-LiteSeg."""

    def __init__(self, channels: int) -> None:
        """Initialize pooling branches.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.scales = (1, 2, 4)
        self.proj = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in self.scales])
        self.fuse = ConvBNReLU(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pooled global context.

        Parameters
        ----------
        x:
            Deep feature map.

        Returns
        -------
        torch.Tensor
            Context-enhanced feature map.
        """

        out = x
        for scale, proj in zip(self.scales, self.proj, strict=True):
            pooled = F.adaptive_avg_pool2d(x, scale)
            out = out + F.interpolate(proj(pooled), size=x.shape[2:], mode="bilinear")
        return self.fuse(out)


class UAFM(nn.Module):
    """Unified Attention Fusion Module."""

    def __init__(self, channels: int) -> None:
        """Initialize attention fusion.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.low_proj = nn.Conv2d(channels, channels, 1)
        self.high_proj = nn.Conv2d(channels, channels, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Fuse low-level and high-level features with learned attention.

        Parameters
        ----------
        low:
            Higher-resolution feature map.
        high:
            Lower-resolution semantic feature map.

        Returns
        -------
        torch.Tensor
            Fused feature map.
        """

        high = F.interpolate(high, size=low.shape[2:], mode="bilinear")
        low = self.low_proj(low)
        high = self.high_proj(high)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(low, 1).expand_as(low),
                F.adaptive_max_pool2d(low, 1).expand_as(low),
                F.adaptive_avg_pool2d(high, 1).expand_as(high),
                F.adaptive_max_pool2d(high, 1).expand_as(high),
            ],
            dim=1,
        )
        weight = self.attn(pooled)
        return low * weight + high * (1.0 - weight)


class CompactPPLiteSegSTDC1(nn.Module):
    """Compact PP-LiteSeg-STDC1 semantic segmenter."""

    def __init__(self, classes: int = 6, width: int = 16) -> None:
        """Initialize the model.

        Parameters
        ----------
        classes:
            Number of semantic classes.
        width:
            Base channel count.
        """

        super().__init__()
        self.stem = ConvBNReLU(3, width, stride=2)
        self.stage2 = STDCBlock(width, width * 2, stride=2)
        self.stage3 = STDCBlock(width * 2, width * 2, stride=2)
        self.stage4 = STDCBlock(width * 2, width * 2, stride=2)
        self.sppm = SPPM(width * 2)
        self.uafm1 = UAFM(width * 2)
        self.uafm2 = UAFM(width * 2)
        self.head = nn.Conv2d(width * 2, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict semantic logits.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Semantic logits at input resolution.
        """

        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.sppm(self.stage4(s3))
        d3 = self.uafm1(s3, s4)
        d2 = self.uafm2(s2, d3)
        return F.interpolate(self.head(d2), size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build the compact PP-LiteSeg-STDC1 model.

    Returns
    -------
    nn.Module
        Random-init segmentation model in evaluation mode.
    """

    return CompactPPLiteSegSTDC1().eval()


def example_input() -> torch.Tensor:
    """Return a compact image batch.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("PP_LiteSeg_STDC1", "build", "example_input", "2022", "DC"),
]
