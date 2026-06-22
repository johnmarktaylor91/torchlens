"""DANet dual-attention semantic segmenter.

Fu et al. (CVPR 2019), "Dual Attention Network for Scene Segmentation."
DANet appends two self-attention modules to a dilated FCN: a position-attention
module aggregates long-range spatially similar features, while a
channel-attention module models dependencies among channel maps.  This compact
version keeps both attention branches and sums their logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        dilation:
            Convolution dilation.
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


class PositionAttention(nn.Module):
    """DANet position-attention module."""

    def __init__(self, channels: int) -> None:
        """Initialize position-attention projections.

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
        """Apply spatial self-attention.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Position-attended features.
        """

        bsz, channels, height, width = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(bsz, channels, height, width)
        return x + self.gamma * out


class ChannelAttention(nn.Module):
    """DANet channel-attention module."""

    def __init__(self) -> None:
        """Initialize the learnable residual scale."""

        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel self-attention.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Channel-attended features.
        """

        bsz, channels, height, width = x.shape
        flat = x.flatten(2)
        energy = torch.bmm(flat, flat.transpose(1, 2))
        attn = torch.softmax(energy.max(dim=-1, keepdim=True).values - energy, dim=-1)
        out = torch.bmm(attn, flat).view(bsz, channels, height, width)
        return x + self.gamma * out


class CompactDANet(nn.Module):
    """Compact DANet with position and channel attention heads."""

    def __init__(self, classes: int = 7, width: int = 16) -> None:
        """Initialize the model.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        width:
            Backbone channel width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNReLU(3, width),
            ConvBNReLU(width, width, dilation=2),
        )
        self.pam = PositionAttention(width)
        self.cam = ChannelAttention()
        self.pam_head = nn.Conv2d(width, classes, 1)
        self.cam_head = nn.Conv2d(width, classes, 1)

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

        feat = self.backbone(F.avg_pool2d(x, 2))
        logits = self.pam_head(self.pam(feat)) + self.cam_head(self.cam(feat))
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build compact DANet.

    Returns
    -------
    nn.Module
        Random-init DANet in evaluation mode.
    """

    return CompactDANet().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("paddleseg_danet", "build", "example_input", "2019", "DC"),
]
