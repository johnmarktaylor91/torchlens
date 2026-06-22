"""OctConv: Octave Convolution.

Paper: Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
Networks with Octave Convolution. Chen et al., ICCV 2019.

OctConv splits feature channels into high- and low-frequency groups.  The low
group is stored at half spatial resolution, while cross-frequency paths exchange
information through average pooling and nearest-neighbor upsampling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OctaveConv2d(nn.Module):
    """Two-frequency convolution with high/low communication paths."""

    def __init__(self, in_channels: int, out_channels: int, alpha: float = 0.5) -> None:
        """Initialize the four OctConv paths.

        Parameters
        ----------
        in_channels:
            Total number of input channels.
        out_channels:
            Total number of output channels.
        alpha:
            Fraction of channels assigned to the low-frequency group.
        """
        super().__init__()
        low_in = int(round(in_channels * alpha))
        low_out = int(round(out_channels * alpha))
        high_in = in_channels - low_in
        high_out = out_channels - low_out
        self.high_out = high_out
        self.low_out = low_out
        self.hh = nn.Conv2d(high_in, high_out, kernel_size=3, padding=1, bias=False)
        self.hl = nn.Conv2d(high_in, low_out, kernel_size=3, padding=1, bias=False)
        self.lh = nn.Conv2d(low_in, high_out, kernel_size=3, padding=1, bias=False)
        self.ll = nn.Conv2d(low_in, low_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x_high: Tensor, x_low: Tensor) -> tuple[Tensor, Tensor]:
        """Apply Octave Convolution to high- and low-frequency tensors.

        Parameters
        ----------
        x_high:
            High-frequency feature map ``(B, C_high, H, W)``.
        x_low:
            Low-frequency feature map ``(B, C_low, H/2, W/2)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated high- and low-frequency feature maps.
        """
        high_to_high = self.hh(x_high)
        high_to_low = self.hl(F.avg_pool2d(x_high, kernel_size=2, stride=2))
        low_to_high = F.interpolate(self.lh(x_low), size=x_high.shape[-2:], mode="nearest")
        low_to_low = self.ll(x_low)
        return high_to_high + low_to_high, low_to_low + high_to_low


class OctaveBlock(nn.Module):
    """OctConv block with per-frequency normalization and activation."""

    def __init__(self, channels: int = 16, alpha: float = 0.5) -> None:
        """Initialize an OctConv processing block.

        Parameters
        ----------
        channels:
            Total feature width across high and low groups.
        alpha:
            Fraction of channels in the low-frequency group.
        """
        super().__init__()
        low_channels = int(round(channels * alpha))
        high_channels = channels - low_channels
        self.octconv = OctaveConv2d(channels, channels, alpha=alpha)
        self.high_norm = nn.BatchNorm2d(high_channels)
        self.low_norm = nn.BatchNorm2d(low_channels)

    def forward(self, x_high: Tensor, x_low: Tensor) -> tuple[Tensor, Tensor]:
        """Run one normalized OctConv block.

        Parameters
        ----------
        x_high:
            High-frequency feature map.
        x_low:
            Low-frequency feature map.

        Returns
        -------
        tuple[Tensor, Tensor]
            Activated high- and low-frequency outputs.
        """
        y_high, y_low = self.octconv(x_high, x_low)
        return torch.relu(self.high_norm(y_high)), torch.relu(self.low_norm(y_low))


class OctConvDemoNet(nn.Module):
    """Compact classifier that preserves separate OctConv frequency streams."""

    def __init__(self, channels: int = 16, num_classes: int = 5, alpha: float = 0.5) -> None:
        """Initialize the OctConv demonstration classifier.

        Parameters
        ----------
        channels:
            Total feature width across both frequency groups.
        num_classes:
            Number of output logits.
        alpha:
            Fraction of channels in the low-frequency group.
        """
        super().__init__()
        low_channels = int(round(channels * alpha))
        high_channels = channels - low_channels
        self.high_stem = nn.Conv2d(3, high_channels, kernel_size=3, padding=1, bias=False)
        self.low_stem = nn.Conv2d(3, low_channels, kernel_size=3, padding=1, bias=False)
        self.block1 = OctaveBlock(channels, alpha=alpha)
        self.block2 = OctaveBlock(channels, alpha=alpha)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits from high/low OctConv streams.

        Parameters
        ----------
        x:
            RGB image tensor with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x_high = torch.relu(self.high_stem(x))
        x_low = torch.relu(self.low_stem(F.avg_pool2d(x, kernel_size=2, stride=2)))
        x_high, x_low = self.block1(x_high, x_low)
        x_high, x_low = self.block2(x_high, x_low)
        low_up = F.interpolate(x_low, size=x_high.shape[-2:], mode="nearest")
        merged = torch.cat([x_high, low_up], dim=1)
        pooled = F.adaptive_avg_pool2d(merged, 1).flatten(1)
        return self.classifier(pooled)


def build() -> nn.Module:
    """Build a compact OctConv demonstration network.

    Returns
    -------
    nn.Module
        Random-initialized OctConv demo network.
    """
    return OctConvDemoNet()


def example_input() -> Tensor:
    """Return a small traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("OctConv (Octave Convolution)", "build", "example_input", "2019", "DC")]
