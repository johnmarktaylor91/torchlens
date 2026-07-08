# SOURCE: vendored from lshiwjx/2s-AGCN @ master: model/agcn.py, graph/ntu_rgb_d.py, graph/tools.py
"""Staged real-source 2s-AGCN model."""

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

MENAGERIE_ZOO = "vendored-pytorch"


def edge2mat(link: list[tuple[int, int]], num_node: int) -> np.ndarray:
    """Build an adjacency matrix from graph links.

    Parameters
    ----------
    link
        Directed graph edges.
    num_node
        Number of graph nodes.

    Returns
    -------
    np.ndarray
        Adjacency matrix.
    """
    a = np.zeros((num_node, num_node))
    for i, j in link:
        a[j, i] = 1
    return a


def normalize_digraph(a: np.ndarray) -> np.ndarray:
    """Normalize a directed graph adjacency matrix.

    Parameters
    ----------
    a
        Adjacency matrix.

    Returns
    -------
    np.ndarray
        Column-normalized adjacency matrix.
    """
    dl = np.sum(a, 0)
    _, w = a.shape
    dn = np.zeros((w, w))
    for i in range(w):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-1)
    return np.dot(a, dn)


def get_spatial_graph(
    num_node: int,
    self_link: list[tuple[int, int]],
    inward: list[tuple[int, int]],
    outward: list[tuple[int, int]],
) -> np.ndarray:
    """Build the spatial graph used by AGCN.

    Parameters
    ----------
    num_node
        Number of graph nodes.
    self_link
        Self-loop edges.
    inward
        Inward skeleton edges.
    outward
        Outward skeleton edges.

    Returns
    -------
    np.ndarray
        Stacked spatial adjacency matrices.
    """
    identity = edge2mat(self_link, num_node)
    in_graph = normalize_digraph(edge2mat(inward, num_node))
    out_graph = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((identity, in_graph, out_graph))


NUM_NODE = 25
SELF_LINK = [(i, i) for i in range(NUM_NODE)]
INWARD_ORI_INDEX = [
    (1, 2),
    (2, 21),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
]
INWARD = [(i - 1, j - 1) for (i, j) in INWARD_ORI_INDEX]
OUTWARD = [(j, i) for (i, j) in INWARD]
NEIGHBOR = INWARD + OUTWARD


class Graph:
    """NTU RGB+D skeleton graph from the upstream 2s-AGCN repository."""

    def __init__(self, labeling_mode: str = "spatial") -> None:
        """Initialize the graph.

        Parameters
        ----------
        labeling_mode
            Graph labeling mode.
        """
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = NUM_NODE
        self.self_link = SELF_LINK
        self.inward = INWARD
        self.outward = OUTWARD
        self.neighbor = NEIGHBOR

    def get_adjacency_matrix(self, labeling_mode: str | None = None) -> np.ndarray:
        """Return the graph adjacency matrix.

        Parameters
        ----------
        labeling_mode
            Graph labeling mode.

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            return get_spatial_graph(NUM_NODE, SELF_LINK, INWARD, OUTWARD)
        raise ValueError


def conv_branch_init(conv: nn.Conv2d, branches: int) -> None:
    """Initialize a branch convolution.

    Parameters
    ----------
    conv
        Convolution module.
    branches
        Number of graph branches.
    """
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv: nn.Conv2d) -> None:
    """Initialize a convolution.

    Parameters
    ----------
    conv
        Convolution module.
    """
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm1d | nn.BatchNorm2d, scale: float) -> None:
    """Initialize a batch normalization module.

    Parameters
    ----------
    bn
        Batch normalization module.
    scale
        Initial scale for the batch-normalization weights.
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class UnitTcn(nn.Module):
    """Temporal convolution unit from upstream AGCN."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1
    ) -> None:
        """Initialize the temporal convolution unit.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Temporal kernel size.
        stride
            Temporal stride.
        """
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the temporal convolution unit.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.bn(self.conv(x))


class UnitGcn(nn.Module):
    """Graph convolution unit from upstream AGCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        a: np.ndarray,
        coff_embedding: int = 4,
        num_subset: int = 3,
    ) -> None:
        """Initialize the graph convolution unit.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        a
            Adjacency matrix.
        coff_embedding
            Embedding reduction coefficient.
        num_subset
            Number of adjacency subsets.
        """
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(a.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(a.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for _ in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down: nn.Module | Any = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the graph convolution unit.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        n, c, t, v = x.size()
        a = self.A.to(device=x.device) + self.PA

        y = None
        for i in range(self.num_subset):
            a1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(n, v, self.inter_c * t)
            a2 = self.conv_b[i](x).view(n, self.inter_c * t, v)
            a1 = self.soft(torch.matmul(a1, a2) / a1.size(-1))
            a1 = a1 + a[i]
            a2 = x.view(n, c * t, v)
            z = self.conv_d[i](torch.matmul(a2, a1).view(n, c, t, v))
            y = z + y if y is not None else z

        if y is None:
            raise RuntimeError("AGCN graph unit produced no branch output")
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TcnGcnUnit(nn.Module):
    """Combined temporal and graph convolution unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        a: np.ndarray,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        """Initialize the combined unit.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        a
            Adjacency matrix.
        stride
            Temporal stride.
        residual
            Whether to include the residual branch.
        """
        super().__init__()
        self.gcn1 = UnitGcn(in_channels, out_channels, a)
        self.tcn1 = UnitTcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the combined unit.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class Model(nn.Module):
    """Adaptive graph convolutional network from 2s-AGCN."""

    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        in_channels: int = 3,
    ) -> None:
        """Initialize the AGCN model.

        Parameters
        ----------
        num_class
            Number of classes.
        num_point
            Number of graph points.
        num_person
            Number of people in each skeleton sequence.
        in_channels
            Number of input channels.
        """
        super().__init__()
        self.graph = Graph()
        a = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TcnGcnUnit(3, 64, a, residual=False)
        self.l2 = TcnGcnUnit(64, 64, a)
        self.l3 = TcnGcnUnit(64, 64, a)
        self.l4 = TcnGcnUnit(64, 64, a)
        self.l5 = TcnGcnUnit(64, 128, a, stride=2)
        self.l6 = TcnGcnUnit(128, 128, a)
        self.l7 = TcnGcnUnit(128, 128, a)
        self.l8 = TcnGcnUnit(128, 256, a, stride=2)
        self.l9 = TcnGcnUnit(256, 256, a)
        self.l10 = TcnGcnUnit(256, 256, a)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the AGCN model.

        Parameters
        ----------
        x
            Skeleton input of shape ``N, C, T, V, M``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        n, c, t, v, m = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        c_new = x.size(1)
        x = x.view(n, m, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)


def build_2s_agcn() -> nn.Module:
    """Build the staged 2s-AGCN model.

    Returns
    -------
    nn.Module
        Model instance.
    """
    return Model(num_class=10)


def example_input_2s_agcn() -> torch.Tensor:
    """Return an example skeleton input.

    Returns
    -------
    torch.Tensor
        Example input tensor.
    """
    return torch.randn(1, 3, 8, 25, 2)


MENAGERIE_ENTRIES = [
    (
        "2s-AGCN",
        "build_2s_agcn",
        "example_input_2s_agcn",
        2019,
        "vendored from lshiwjx/2s-AGCN model/agcn.py",
    ),
]
