"""DARTS search CNN supernet, 2019.

Paper: DARTS: Differentiable Architecture Search (Liu, Simonyan, Yang; ICLR 2019).

Compact random-init search-stage supernet: every edge in a four-node cell is a
softmax-weighted mixed operation over the DARTS primitive set (none, pooling,
skip, separable conv, dilated conv), with separate normal and reduction cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from menagerie.classics.darts_cifar_derived_network import FactorizedReduce, ReLUConvBN, _op


PRIMITIVES: tuple[str, ...] = (
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
)


class Zero(nn.Module):
    """DARTS none operation, with optional stride."""

    def __init__(self, stride: int) -> None:
        """Create a zero edge.

        Parameters
        ----------
        stride
            Spatial stride to mimic for shape compatibility.
        """
        super().__init__()
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Return a zero tensor with the edge output shape.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Zero-valued output.
        """
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class MixedOp(nn.Module):
    """Softmax-weighted mixture over the DARTS primitive operation set."""

    def __init__(self, channels: int, stride: int) -> None:
        """Create a mixed operation edge.

        Parameters
        ----------
        channels
            Edge channel count.
        stride
            Edge stride.
        """
        super().__init__()
        self.ops = nn.ModuleList(
            [
                Zero(stride) if primitive == "none" else _op(primitive, channels, stride)
                for primitive in PRIMITIVES
            ]
        )
        self.alpha = nn.Parameter(torch.zeros(len(PRIMITIVES)))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the weighted primitive mixture.

        Parameters
        ----------
        x
            Edge input.

        Returns
        -------
        Tensor
            Weighted mixed edge output.
        """
        weights = torch.softmax(self.alpha, dim=0)
        out = self.ops[0](x) * weights[0]
        for idx, op in enumerate(self.ops[1:], start=1):
            out = out + op(x) * weights[idx]
        return out


class SearchCell(nn.Module):
    """DARTS continuous-relaxation cell with all previous-node edges."""

    def __init__(
        self,
        channels_prev_prev: int,
        channels_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
        steps: int = 4,
    ) -> None:
        """Create a search cell.

        Parameters
        ----------
        channels_prev_prev
            Channels from cell ``k - 2``.
        channels_prev
            Channels from cell ``k - 1``.
        channels
            Internal channel count.
        reduction
            Whether this cell downsamples input-node edges.
        reduction_prev
            Whether the previous cell reduced spatial size.
        steps
            Number of intermediate nodes. DARTS search cells use four.
        """
        super().__init__()
        self.pre0 = (
            FactorizedReduce(channels_prev_prev, channels)
            if reduction_prev
            else ReLUConvBN(channels_prev_prev, channels, 1, 1, 0)
        )
        self.pre1 = ReLUConvBN(channels_prev, channels, 1, 1, 0)
        self.steps = steps
        self.edges = nn.ModuleList()
        for node in range(steps):
            for src in range(node + 2):
                stride = 2 if reduction and src < 2 else 1
                self.edges.append(MixedOp(channels, stride))

    def forward(self, prev_prev: Tensor, prev: Tensor) -> Tensor:
        """Run the densely connected mixed-operation cell.

        Parameters
        ----------
        prev_prev
            State from cell ``k - 2``.
        prev
            State from cell ``k - 1``.

        Returns
        -------
        Tensor
            Concatenated intermediate node states.
        """
        states = [self.pre0(prev_prev), self.pre1(prev)]
        offset = 0
        for node in range(self.steps):
            node_sum = self.edges[offset](states[0])
            for src in range(1, node + 2):
                node_sum = node_sum + self.edges[offset + src](states[src])
            offset += node + 2
            states.append(node_sum)
        return torch.cat(states[2:], dim=1)


class DARTSSearchCNN(nn.Module):
    """Compact DARTS search-stage CNN supernet."""

    def __init__(self, channels: int = 4, classes: int = 10, layers: int = 3) -> None:
        """Create the search supernet.

        Parameters
        ----------
        channels
            Initial channel count.
        classes
            Number of output classes.
        layers
            Number of search cells.
        """
        super().__init__()
        stem_ch = 3 * channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
        )
        c_prev_prev = stem_ch
        c_prev = stem_ch
        c_curr = channels
        reduction_prev = False
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = i in {layers // 3, 2 * layers // 3}
            if reduction:
                c_curr *= 2
            cell = SearchCell(c_prev_prev, c_prev, c_curr, reduction, reduction_prev, steps=4)
            self.cells.append(cell)
            reduction_prev = reduction
            c_prev_prev, c_prev = c_prev, 4 * c_curr
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image through mixed-operation cells.

        Parameters
        ----------
        x
            Input image batch.

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
    """Build the compact DARTS search CNN supernet.

    Returns
    -------
    nn.Module
        Random-init DARTS search supernet.
    """
    return DARTSSearchCNN().eval()


def example_input() -> Tensor:
    """Return a CIFAR-sized search input.

    Returns
    -------
    Tensor
        Example image batch.
    """
    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = [
    ("DARTS search CNN supernet", "build", "example_input", "2019", "E7"),
]
