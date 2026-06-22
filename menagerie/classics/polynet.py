"""PolyNet: very deep PolyInception CNN for image recognition.

Paper: PolyNet: A Pursuit of Structural Diversity in Very Deep Networks.
Zhang, Li, Loy, and Lin, CVPR 2017.

The published model replaces some Inception-ResNet blocks with PolyInception
modules: residual Inception functions are composed as polynomial terms, e.g.
``x + F(x) + G(F(x))`` or longer chains, giving structural diversity without
only making the network deeper or wider.  This compact random-init classic keeps
the load-bearing primitive: Inception-style residual branches arranged as first,
second, and third-order polynomial compositions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    """Convolution, batch normalization, and GELU activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
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
            nn.GELU(),
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


class InceptionResidual(nn.Module):
    """Compact Inception-ResNet residual function used inside PolyInception."""

    def __init__(self, channels: int, branch_channels: int = 8, scale: float = 0.2) -> None:
        """Initialize the residual Inception function.

        Parameters
        ----------
        channels:
            Input and output channel count.
        branch_channels:
            Width of each Inception branch.
        scale:
            Residual scaling factor.
        """

        super().__init__()
        self.scale = scale
        self.branch_1 = ConvBnAct(channels, branch_channels, 1)
        self.branch_3 = nn.Sequential(
            ConvBnAct(channels, branch_channels, 1),
            ConvBnAct(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch_5 = nn.Sequential(
            ConvBnAct(channels, branch_channels, 1),
            ConvBnAct(branch_channels, branch_channels, 3, padding=1),
            ConvBnAct(branch_channels, branch_channels, 3, padding=1),
        )
        self.mix = nn.Conv2d(branch_channels * 3, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate one scaled Inception residual function.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual update with the same shape as ``x``.
        """

        branches = torch.cat((self.branch_1(x), self.branch_3(x), self.branch_5(x)), dim=1)
        return self.scale * self.bn(self.mix(branches))


class PolyInception(nn.Module):
    """Polynomial composition of residual Inception functions."""

    def __init__(self, channels: int, order: int) -> None:
        """Initialize a PolyInception module.

        Parameters
        ----------
        channels:
            Input and output channel count.
        order:
            Polynomial order, from one to three in this compact model.
        """

        super().__init__()
        if order < 1:
            msg = "PolyInception order must be at least one."
            raise ValueError(msg)
        self.functions = nn.ModuleList([InceptionResidual(channels) for _ in range(order)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply first and higher-order residual polynomial terms.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Feature map after polynomial residual composition.
        """

        out = x
        term = x
        for fn in self.functions:
            term = fn(term)
            out = out + term
        return F.gelu(out)


class CompactPolyNet(nn.Module):
    """Small PolyNet image classifier with stacked PolyInception modules."""

    def __init__(self, num_classes: int = 10, channels: int = 32) -> None:
        """Initialize the compact PolyNet.

        Parameters
        ----------
        num_classes:
            Number of classifier outputs.
        channels:
            Internal feature width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct(3, channels // 2, 3, stride=2, padding=1),
            ConvBnAct(channels // 2, channels, 3, padding=1),
        )
        self.poly_1 = PolyInception(channels, order=1)
        self.reduction = ConvBnAct(channels, channels, 3, stride=2, padding=1)
        self.poly_2 = PolyInception(channels, order=2)
        self.poly_3 = PolyInception(channels, order=3)
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
        x = self.poly_1(x)
        x = self.reduction(x)
        x = self.poly_2(x)
        x = self.poly_3(x)
        return self.head(self.pool(x).flatten(1))


def build() -> nn.Module:
    """Build the compact PolyNet classic.

    Returns
    -------
    nn.Module
        Evaluation-mode compact PolyNet model.
    """

    return CompactPolyNet().eval()


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
        "PolyNet (Very Deep PolyInception CNN)",
        "build",
        "example_input",
        "2017",
        "E5",
    ),
]
