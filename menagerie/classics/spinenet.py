"""SpineNet: scale-permuted backbone for recognition and localization.

Paper: SpineNet: Learning Scale-Permuted Backbone for Recognition and
Localization. Du et al., CVPR 2020.

SpineNet rejects the usual strictly scale-decreasing backbone.  It uses a
scale-permuted sequence of residual blocks where each block resamples and fuses
two earlier feature levels, producing strong multi-scale features for detection.
This compact random-init model keeps the scale permutation and cross-scale
resampling/fusion topology while reducing the block count and width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Convolution, batch normalization, and ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
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


class ResampleFeature(nn.Module):
    """Project a feature map and resample it to a target pyramid level."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the resampling projection.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        """

        super().__init__()
        self.proj = ConvBnRelu(in_channels, out_channels, 1, padding=0)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Project and resize a feature map.

        Parameters
        ----------
        x:
            Input feature map.
        size:
            Target spatial ``(height, width)``.

        Returns
        -------
        torch.Tensor
            Projected feature map at the target resolution.
        """

        x = self.proj(x)
        if x.shape[2:] == size:
            return x
        if x.shape[2] > size[0]:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")


class SpineBlock(nn.Module):
    """Scale-permuted block that fuses two parent feature levels."""

    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int,
        out_channels: int,
        target_size: tuple[int, int],
    ) -> None:
        """Initialize a SpineNet fusion block.

        Parameters
        ----------
        in_channels_a:
            Channel count for the first parent.
        in_channels_b:
            Channel count for the second parent.
        out_channels:
            Output channel count.
        target_size:
            Target spatial resolution for this scale level.
        """

        super().__init__()
        self.target_size = target_size
        self.resample_a = ResampleFeature(in_channels_a, out_channels)
        self.resample_b = ResampleFeature(in_channels_b, out_channels)
        self.block = nn.Sequential(
            ConvBnRelu(out_channels, out_channels),
            ConvBnRelu(out_channels, out_channels),
        )

    def forward(self, parent_a: torch.Tensor, parent_b: torch.Tensor) -> torch.Tensor:
        """Fuse two parent features at the target scale.

        Parameters
        ----------
        parent_a:
            First parent feature map.
        parent_b:
            Second parent feature map.

        Returns
        -------
        torch.Tensor
            Scale-permuted block output.
        """

        fused = self.resample_a(parent_a, self.target_size) + self.resample_b(
            parent_b,
            self.target_size,
        )
        return self.block(fused)


class CompactSpineNet(nn.Module):
    """Small SpineNet-style scale-permuted backbone."""

    def __init__(self, num_classes: int = 10, channels: int = 24) -> None:
        """Initialize the compact SpineNet.

        Parameters
        ----------
        num_classes:
            Number of classifier outputs.
        channels:
            Internal feature width.
        """

        super().__init__()
        self.stem_2 = ConvBnRelu(3, channels, stride=2)
        self.stem_3 = ConvBnRelu(channels, channels, stride=2)
        self.block_4a = SpineBlock(channels, channels, channels, (4, 4))
        self.block_3a = SpineBlock(channels, channels, channels, (8, 8))
        self.block_5a = SpineBlock(channels, channels, channels, (2, 2))
        self.block_4b = SpineBlock(channels, channels, channels, (4, 4))
        self.block_6a = SpineBlock(channels, channels, channels, (1, 1))
        self.fuse = ConvBnRelu(channels * 3, channels, 1, padding=0)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Image tensor with shape ``(B, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Class logits from fused scale-permuted features.
        """

        level_2 = self.stem_2(x)
        level_3 = self.stem_3(level_2)
        level_4a = self.block_4a(level_2, level_3)
        level_3a = self.block_3a(level_3, level_4a)
        level_5a = self.block_5a(level_4a, level_3a)
        level_4b = self.block_4b(level_5a, level_3a)
        level_6a = self.block_6a(level_4b, level_5a)
        fused = torch.cat(
            (
                F.adaptive_avg_pool2d(level_3a, (1, 1)),
                F.adaptive_avg_pool2d(level_4b, (1, 1)),
                level_6a,
            ),
            dim=1,
        )
        fused = self.fuse(fused).flatten(1)
        return self.head(fused)


def build() -> nn.Module:
    """Build the compact SpineNet classic.

    Returns
    -------
    nn.Module
        Evaluation-mode compact SpineNet model.
    """

    return CompactSpineNet().eval()


def example_input() -> torch.Tensor:
    """Create the canonical example image batch.

    Returns
    -------
    torch.Tensor
        Random image tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SpineNet (scale-permuted detection backbone)",
        "build",
        "example_input",
        "2020",
        "E5",
    ),
]
