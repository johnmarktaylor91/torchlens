"""Big-Little Net: efficient multi-scale CNN with frequent branch merging.

Paper: Big-Little Net: An Efficient Multi-Scale Feature Representation for
Visual and Speech Recognition. Chen, Fan, Mallinar, Sercu, and Feris, ICLR 2019.

bL-Net uses paired branches at different image scales.  The Big branch keeps the
baseline high-capacity computation on lower-resolution features; the Little
branch runs cheaper convolutions at higher resolution.  Frequent merge operations
exchange information across branches.  This compact version keeps the same
two-branch, repeated-merge topology with reduced channels and random weights.
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
        groups: int = 1,
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
        groups:
            Number of convolution groups.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
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


class BigLittleModule(nn.Module):
    """Two-scale bL module with cheap high-resolution and rich low-resolution branches."""

    def __init__(self, channels: int, little_ratio: float = 0.5) -> None:
        """Initialize a Big-Little module.

        Parameters
        ----------
        channels:
            Big-branch channel count.
        little_ratio:
            Fraction of channels assigned to the Little branch.
        """

        super().__init__()
        little_channels = max(4, int(channels * little_ratio))
        self.big = nn.Sequential(
            ConvBnRelu(channels, channels),
            ConvBnRelu(channels, channels),
        )
        self.little_reduce = ConvBnRelu(channels, little_channels, 1, padding=0)
        self.little_depthwise = ConvBnRelu(
            little_channels,
            little_channels,
            groups=little_channels,
        )
        self.little_expand = ConvBnRelu(little_channels, channels, 1, padding=0)
        self.big_to_little = nn.Conv2d(channels, channels, 1, bias=False)
        self.little_to_big = nn.Conv2d(channels, channels, 1, bias=False)
        self.out = ConvBnRelu(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one frequent merge between Big and Little branches.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Merged feature map with the same spatial size as ``x``.
        """

        big_in = F.avg_pool2d(x, kernel_size=2, stride=2)
        little = self.little_expand(self.little_depthwise(self.little_reduce(x)))
        big = self.big(big_in)
        little_from_big = F.interpolate(
            self.big_to_little(big),
            size=little.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        big_from_little = self.little_to_big(F.avg_pool2d(little, kernel_size=2, stride=2))
        big = big + big_from_little
        little = little + little_from_big
        return self.out(F.interpolate(big, size=x.shape[2:], mode="nearest") + little)


class CompactBigLittleNet(nn.Module):
    """Compact image classifier built from stacked Big-Little modules."""

    def __init__(self, num_classes: int = 10, channels: int = 32) -> None:
        """Initialize the compact bL-Net.

        Parameters
        ----------
        num_classes:
            Number of classifier outputs.
        channels:
            Internal feature width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(3, channels, stride=2),
            ConvBnRelu(channels, channels),
        )
        self.blocks = nn.Sequential(
            BigLittleModule(channels),
            ConvBnRelu(channels, channels, stride=2),
            BigLittleModule(channels),
            BigLittleModule(channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
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
            Class logits with shape ``(B, num_classes)``.
        """

        x = self.stem(x)
        x = self.blocks(x)
        return self.head(self.pool(x).flatten(1))


def build() -> nn.Module:
    """Build the compact Big-Little Net classic.

    Returns
    -------
    nn.Module
        Evaluation-mode compact bL-Net model.
    """

    return CompactBigLittleNet().eval()


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
        "Big-Little Net (bL-Net multi-scale CNN)",
        "build",
        "example_input",
        "2019",
        "E5",
    ),
]
