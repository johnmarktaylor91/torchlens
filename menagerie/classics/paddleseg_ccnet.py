"""CCNet criss-cross-attention semantic segmenter.

Huang et al. (ICCV 2019), "CCNet: Criss-Cross Attention for Semantic
Segmentation."  CCNet harvests context along each pixel's horizontal and
vertical criss-cross path, then repeats the operation recurrently so every
pixel can receive full-image context without full non-local attention cost.
This compact reconstruction keeps the recurrent criss-cross attention (RCCA)
primitive and a small segmentation head.
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


class CrissCrossAttention(nn.Module):
    """Attention over horizontal and vertical paths through each pixel."""

    def __init__(self, channels: int) -> None:
        """Initialize criss-cross projections.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        hidden = max(4, channels // 2)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one criss-cross attention pass.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        bsz, channels, height, width = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        row_scores = torch.einsum("bchw,bcvw->bhwv", q, k) / (q.shape[1] ** 0.5)
        col_scores = torch.einsum("bchw,bchu->bhwu", q, k) / (q.shape[1] ** 0.5)
        row_attn = torch.softmax(row_scores, dim=-1)
        col_attn = torch.softmax(col_scores, dim=-1)
        row_out = torch.einsum("bhwv,bcvw->bchw", row_attn, v)
        col_out = torch.einsum("bhwu,bchu->bchw", col_attn, v)
        return x + self.gamma * (row_out + col_out).view(bsz, channels, height, width)


class CompactCCNet(nn.Module):
    """Compact CCNet with recurrent criss-cross attention."""

    def __init__(self, classes: int = 7, width: int = 16) -> None:
        """Initialize the compact model.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        width:
            Base channel width.
        """

        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, width, 2), ConvBNReLU(width, width, 1))
        self.cca = CrissCrossAttention(width)
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

        feat = self.stem(x)
        feat = self.cca(self.cca(feat))
        return F.interpolate(self.head(feat), size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build compact CCNet.

    Returns
    -------
    nn.Module
        Random-init CCNet in evaluation mode.
    """

    return CompactCCNet().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("paddleseg_ccnet", "build", "example_input", "2019", "DC"),
]
