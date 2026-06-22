"""DARTS CIFAR derived CNN, 2019.

Paper: DARTS: Differentiable Architecture Search (Liu, Simonyan, Yang; ICLR 2019).

Faithful compact random-init reconstruction of the evaluation-time CIFAR network:
stacked learned normal/reduction cells, two previous cell states as inputs, four
intermediate nodes per cell, two selected edges per node, channel-concat output,
factorized reduction for stride-2 skip edges, and the published DARTS_V2 genotype.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Genotype = tuple[
    tuple[tuple[str, int], ...],
    tuple[int, ...],
    tuple[tuple[str, int], ...],
    tuple[int, ...],
]

DARTS_V2: Genotype = (
    (
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("dil_conv_3x3", 2),
    ),
    (2, 3, 4, 5),
    (
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
    ),
    (2, 3, 4, 5),
)


class ReLUConvBN(nn.Sequential):
    """DARTS ReLU-conv-batchnorm primitive."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> None:
        """Create the primitive.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        kernel
            Convolution kernel size.
        stride
            Convolution stride.
        padding
            Convolution padding.
        """
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
        )


class FactorizedReduce(nn.Module):
    """DARTS stride-2 skip connection that concatenates two offset 1x1 paths."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Create a factorized reduction.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        """
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch - out_ch // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Reduce spatial size and adjust channels.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Reduced feature map.
        """
        x = self.relu(x)
        return self.bn(torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1))


class SepConv(nn.Sequential):
    """Two DARTS depthwise-separable convolution stages."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> None:
        """Create a separable convolution primitive.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        kernel
            Depthwise kernel size.
        stride
            First depthwise stride.
        padding
            Depthwise padding.
        """
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_ch, in_ch, kernel, stride=stride, padding=padding, groups=in_ch, bias=False
            ),
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )


class DilConv(nn.Sequential):
    """DARTS dilated depthwise-separable convolution primitive."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int, dilation: int
    ) -> None:
        """Create a dilated separable convolution.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        kernel
            Depthwise kernel size.
        stride
            Depthwise stride.
        padding
            Depthwise padding.
        dilation
            Depthwise dilation.
        """
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )


class Identity(nn.Module):
    """Trace-friendly identity module."""

    def forward(self, x: Tensor) -> Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Tensor
            Same tensor value.
        """
        return x


def _op(name: str, channels: int, stride: int) -> nn.Module:
    """Build a DARTS operation by primitive name.

    Parameters
    ----------
    name
        DARTS primitive name.
    channels
        Input and output channels.
    stride
        Operation stride.

    Returns
    -------
    nn.Module
        Primitive module.
    """
    if name == "max_pool_3x3":
        return nn.MaxPool2d(3, stride=stride, padding=1)
    if name == "avg_pool_3x3":
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    if name == "skip_connect":
        return Identity() if stride == 1 else FactorizedReduce(channels, channels)
    if name == "sep_conv_3x3":
        return SepConv(channels, channels, 3, stride, 1)
    if name == "sep_conv_5x5":
        return SepConv(channels, channels, 5, stride, 2)
    if name == "dil_conv_3x3":
        return DilConv(channels, channels, 3, stride, 2, 2)
    if name == "dil_conv_5x5":
        return DilConv(channels, channels, 5, stride, 4, 2)
    raise ValueError(f"Unsupported DARTS op: {name}")


class DARTSCell(nn.Module):
    """DARTS genotype cell with two incoming selected edges per node."""

    def __init__(
        self,
        genotype: Genotype,
        channels_prev_prev: int,
        channels_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        """Create a learned DARTS cell.

        Parameters
        ----------
        genotype
            DARTS normal/reduction genotype.
        channels_prev_prev
            Channels from cell ``k - 2``.
        channels_prev
            Channels from cell ``k - 1``.
        channels
            Internal cell channels.
        reduction
            Whether this cell downsamples.
        reduction_prev
            Whether the previous cell was a reduction cell.
        """
        super().__init__()
        self.pre0 = (
            FactorizedReduce(channels_prev_prev, channels)
            if reduction_prev
            else ReLUConvBN(channels_prev_prev, channels, 1, 1, 0)
        )
        self.pre1 = ReLUConvBN(channels_prev, channels, 1, 1, 0)
        genes, concat = (genotype[2], genotype[3]) if reduction else (genotype[0], genotype[1])
        self.concat = concat
        self.indices = [idx for _, idx in genes]
        self.ops = nn.ModuleList(
            [_op(name, channels, 2 if reduction and idx < 2 else 1) for name, idx in genes]
        )

    def forward(self, prev_prev: Tensor, prev: Tensor) -> Tensor:
        """Run the two-input cell DAG.

        Parameters
        ----------
        prev_prev
            State from cell ``k - 2``.
        prev
            State from cell ``k - 1``.

        Returns
        -------
        Tensor
            Concatenated selected intermediate states.
        """
        states = [self.pre0(prev_prev), self.pre1(prev)]
        for step in range(len(self.ops) // 2):
            left = self.ops[2 * step](states[self.indices[2 * step]])
            right = self.ops[2 * step + 1](states[self.indices[2 * step + 1]])
            states.append(left + right)
        return torch.cat([states[i] for i in self.concat], dim=1)


class DARTSCifarNetwork(nn.Module):
    """Compact CIFAR DARTS evaluation network with normal/reduction cells."""

    def __init__(self, channels: int = 4, classes: int = 10, layers: int = 3) -> None:
        """Create a compact CIFAR classifier.

        Parameters
        ----------
        channels
            Initial cell channel count.
        classes
            Number of output classes.
        layers
            Number of DARTS cells.
        """
        super().__init__()
        stem_channels = 3 * channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
        )
        c_prev_prev = stem_channels
        c_prev = stem_channels
        c_curr = channels
        reduction_prev = False
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = i in {layers // 3, 2 * layers // 3}
            if reduction:
                c_curr *= 2
            cell = DARTSCell(DARTS_V2, c_prev_prev, c_prev, c_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            c_prev_prev, c_prev = c_prev, len(DARTS_V2[1]) * c_curr
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image.

        Parameters
        ----------
        x
            Input image batch ``(B, 3, 32, 32)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        s0 = self.stem(x)
        s1 = s0
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return self.classifier(torch.flatten(self.pool(s1), 1))


def build() -> nn.Module:
    """Build the compact DARTS CIFAR derived network.

    Returns
    -------
    nn.Module
        Random-init DARTS CIFAR network.
    """
    return DARTSCifarNetwork().eval()


def example_input() -> Tensor:
    """Return a CIFAR-sized image input.

    Returns
    -------
    Tensor
        Example image batch.
    """
    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = [
    ("DARTS CIFAR derived network", "build", "example_input", "2019", "E7"),
]
