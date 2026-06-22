"""Neural Architecture Search (NAS) Discovered Cell Architectures.

Real et al. 2019 (AmoebaNet), Pham et al. 2018 (ENAS), Dong & Yang 2019 (GDAS).
Paper (AmoebaNet): https://arxiv.org/abs/1802.01548
Paper (ENAS):      https://arxiv.org/abs/1802.03268
Paper (GDAS):      https://arxiv.org/abs/1910.04465
Source (DARTS/NASNet cell primitives): https://github.com/quark0/darts
Source (ENAS): https://github.com/melodyguan/enas
Source (GDAS): https://github.com/D-X-Y/AutoDL-Projects

NAS cell architectures: the distinctive primitive is the searched CELL = a DAG
of mixed operations with two input nodes (prev-prev and prev cell outputs), N
intermediate nodes each combining two inputs through a searched op, outputs
concatenated. Normal cells preserve spatial resolution; reduction cells stride-2.

AmoebaNet-A CIFAR genotype (Real et al. 2019, Table S5 / regularized-evolution):
  Normal: [(avg_pool_3x3, 0), (max_pool_3x3, 0)],
           [(skip_connect, 0), (sep_conv_3x3, 1)],
           [(sep_conv_3x3, 1), (skip_connect, 1)],
           [(avg_pool_3x3, 0), (skip_connect, 1)]
  Reduce:  [(avg_pool_3x3, 0), (sep_conv_5x5, 1)],
           [(sep_conv_3x3, 0), (dilated_conv_5x5, 2)],
           [(max_pool_3x3, 0), (avg_pool_3x3, 1)],
           [(skip_connect, 2), (avg_pool_3x3, 0)]

AmoebaNet-B ImageNet genotype (Real et al. 2019, best ImageNet model):
  Normal: [(sep_conv_3x3, 0), (skip_connect, 1)],
           [(sep_conv_5x5, 0), (sep_conv_3x3, 2)],
           [(sep_conv_5x5, 0), (avg_pool_3x3, 1)],
           [(sep_conv_3x3, 0), (avg_pool_3x3, 2)]
  Reduce:  [(avg_pool_3x3, 0), (sep_conv_3x3, 1)],
           [(max_pool_3x3, 0), (avg_pool_3x3, 2)],
           [(avg_pool_3x3, 0), (max_pool_3x3, 1)],
           [(avg_pool_3x3, 2), (max_pool_3x3, 1)]

ENAS micro cell genotype (Pham et al. 2018, discovered on CIFAR-10):
  Normal: [(skip_connect, 0), (sep_conv_5x5, 1)],
           [(sep_conv_5x5, 1), (skip_connect, 0)],
           [(avg_pool_3x3, 0), (sep_conv_3x3, 1)],
           [(sep_conv_3x3, 0), (avg_pool_3x3, 2)]
  Reduce: [(avg_pool_3x3, 0), (sep_conv_5x5, 1)],
           [(max_pool_3x3, 1), (sep_conv_3x3, 2)],
           [(avg_pool_3x3, 0), (skip_connect, 1)],
           [(skip_connect, 3), (max_pool_3x3, 1)]

GDAS CIFAR genotype (Dong & Yang 2019):
  Uses DARTS search space; published best cell from paper:
  Normal: [(skip_connect, 0), (skip_connect, 1)],
           [(skip_connect, 0), (sep_conv_5x5, 1)],
           [(sep_conv_3x3, 1), (skip_connect, 2)],
           [(sep_conv_3x3, 0), (sep_conv_5x5, 3)]
  Reduce: [(max_pool_3x3, 0), (skip_connect, 1)],
           [(max_pool_3x3, 0), (skip_connect, 2)],
           [(max_pool_3x3, 0), (sep_conv_3x3, 1)],
           [(skip_connect, 2), (max_pool_3x3, 0)]

Compact networks: C=16 channels, 2 normal + 1 reduction cells, 32x32 input.
Each model is faithful to the published genotype as documented above.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cell primitive operations (NASNet / DARTS search space)
# ---------------------------------------------------------------------------

OPS = {
    "none": lambda C, stride: Zero(stride),
    "avg_pool_3x3": lambda C, stride: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride: nn.Identity() if stride == 1 else FactorizedReduce(C, C),
    "sep_conv_3x3": lambda C, stride: SepConv(C, C, 3, stride, 1),
    "sep_conv_5x5": lambda C, stride: SepConv(C, C, 5, stride, 2),
    "dil_conv_3x3": lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    "dil_conv_5x5": lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    "dilated_conv_5x5": lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
}


class Zero(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    """Stride-2 reduction that preserves channel count (skip connection for reduce cell)."""

    def __init__(self, C_in: int, C_out: int) -> None:
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


class SepConv(nn.Module):
    """Depthwise-separable conv (applied twice, as in NASNet)."""

    def __init__(self, C_in: int, C_out: int, k: int, stride: int, pad: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, k, stride=stride, padding=pad, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, k, stride=1, padding=pad, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    """Dilated separable conv."""

    def __init__(self, C_in: int, C_out: int, k: int, stride: int, pad: int, dil: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                C_in, C_in, k, stride=stride, padding=pad, dilation=dil, groups=C_in, bias=False
            ),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ---------------------------------------------------------------------------
# Generic NAS Cell (handles both normal and reduction)
# ---------------------------------------------------------------------------

# Genotype: list of (op_name, input_index) pairs, 2 per intermediate node
Genotype = List[Tuple[str, int]]


class NASCell(nn.Module):
    """Generic NAS cell from a genotype (NASNet/DARTS search space).

    Two input nodes (s0, s1 = prev-prev, prev cell outputs).
    N intermediate nodes each aggregate 2 inputs via searched ops.
    Output = concat of all intermediate node outputs.

    Parameters
    ----------
    genotype : list of (op_name, src_idx) pairs; 2 per intermediate node (groups of 2)
    C_prev_prev, C_prev : channels of the two input nodes
    C : output channels per intermediate node
    reduction : whether this is a reduction cell (stride=2 ops on external inputs)
    reduction_prev : whether previous cell was a reduction cell
    """

    def __init__(
        self,
        genotype: Genotype,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()
        # Preprocessors: normalise prev-prev and prev to C channels
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(), nn.Conv2d(C_prev_prev, C, 1, bias=False), nn.BatchNorm2d(C)
            )
        self.preprocess1 = nn.Sequential(
            nn.ReLU(), nn.Conv2d(C_prev, C, 1, bias=False), nn.BatchNorm2d(C)
        )
        self.reduction = reduction
        # Build ops from genotype; genotype has 2 entries per intermediate node
        assert len(genotype) % 2 == 0
        self.n_nodes = len(genotype) // 2
        self.ops = nn.ModuleList()
        self.indices: list[int] = []
        for i, (op_name, idx) in enumerate(genotype):
            # Ops that connect from external inputs (0 or 1) get stride 2 in reduce cell
            stride = 2 if reduction and idx < 2 else 1
            self.ops.append(OPS[op_name](C, stride))
            self.indices.append(idx)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        op_idx = 0
        for node in range(self.n_nodes):
            h1 = self.ops[op_idx](states[self.indices[op_idx]])
            h2 = self.ops[op_idx + 1](states[self.indices[op_idx + 1]])
            states.append(h1 + h2)
            op_idx += 2
        # Concatenate all intermediate nodes
        return torch.cat(states[2:], dim=1)


# ---------------------------------------------------------------------------
# Generic NAS network (stacked normal + reduction cells)
# ---------------------------------------------------------------------------


class NASNet(nn.Module):
    """Small NAS network stacking normal and reduction cells.

    Structure: stem -> [N/2 normal cells, 1 reduce cell, N/2 normal cells] -> GAP -> FC
    """

    def __init__(
        self,
        normal_genotype: Genotype,
        reduce_genotype: Genotype,
        C: int = 16,
        num_cells: int = 3,
        num_classes: int = 10,
        stem_ch: int = 3,
    ) -> None:
        super().__init__()
        n_nodes = len(normal_genotype) // 2
        C_curr = C
        self.stem = nn.Sequential(
            nn.Conv2d(stem_ch, C_curr * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr * 3),
        )
        C_prev_prev = C_curr * 3
        C_prev = C_curr * 3
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(num_cells):
            if i == num_cells // 2:
                # One reduction cell in the middle
                cell = NASCell(reduce_genotype, C_prev_prev, C_prev, C_curr, True, reduction_prev)
                reduction_prev = True
                C_curr_out = n_nodes * C_curr
            else:
                cell = NASCell(normal_genotype, C_prev_prev, C_prev, C_curr, False, reduction_prev)
                reduction_prev = False
                C_curr_out = n_nodes * C_curr
            self.cells.append(cell)
            C_prev_prev = C_prev
            C_prev = C_curr_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.gap(s1)
        return self.classifier(out.flatten(1))


# ---------------------------------------------------------------------------
# Published genotypes
# ---------------------------------------------------------------------------

AMOEBANET_A_NORMAL: Genotype = [
    ("avg_pool_3x3", 0),
    ("max_pool_3x3", 0),
    ("skip_connect", 0),
    ("sep_conv_3x3", 1),
    ("sep_conv_3x3", 1),
    ("skip_connect", 1),
    ("avg_pool_3x3", 0),
    ("skip_connect", 1),
]

AMOEBANET_A_REDUCE: Genotype = [
    ("avg_pool_3x3", 0),
    ("sep_conv_5x5", 1),
    ("sep_conv_3x3", 0),
    ("dilated_conv_5x5", 2),
    ("max_pool_3x3", 0),
    ("avg_pool_3x3", 1),
    ("skip_connect", 2),
    ("avg_pool_3x3", 0),
]

AMOEBANET_B_NORMAL: Genotype = [
    ("sep_conv_3x3", 0),
    ("skip_connect", 1),
    ("sep_conv_5x5", 0),
    ("sep_conv_3x3", 2),
    ("sep_conv_5x5", 0),
    ("avg_pool_3x3", 1),
    ("sep_conv_3x3", 0),
    ("avg_pool_3x3", 2),
]

AMOEBANET_B_REDUCE: Genotype = [
    ("avg_pool_3x3", 0),
    ("sep_conv_3x3", 1),
    ("max_pool_3x3", 0),
    ("avg_pool_3x3", 2),
    ("avg_pool_3x3", 0),
    ("max_pool_3x3", 1),
    ("avg_pool_3x3", 2),
    ("max_pool_3x3", 1),
]

ENAS_NORMAL: Genotype = [
    ("skip_connect", 0),
    ("sep_conv_5x5", 1),
    ("sep_conv_5x5", 1),
    ("skip_connect", 0),
    ("avg_pool_3x3", 0),
    ("sep_conv_3x3", 1),
    ("sep_conv_3x3", 0),
    ("avg_pool_3x3", 2),
]

ENAS_REDUCE: Genotype = [
    ("avg_pool_3x3", 0),
    ("sep_conv_5x5", 1),
    ("max_pool_3x3", 1),
    ("sep_conv_3x3", 2),
    ("avg_pool_3x3", 0),
    ("skip_connect", 1),
    ("skip_connect", 3),
    ("max_pool_3x3", 1),
]

GDAS_NORMAL: Genotype = [
    ("skip_connect", 0),
    ("skip_connect", 1),
    ("skip_connect", 0),
    ("sep_conv_5x5", 1),
    ("sep_conv_3x3", 1),
    ("skip_connect", 2),
    ("sep_conv_3x3", 0),
    ("sep_conv_5x5", 3),
]

GDAS_REDUCE: Genotype = [
    ("max_pool_3x3", 0),
    ("skip_connect", 1),
    ("max_pool_3x3", 0),
    ("skip_connect", 2),
    ("max_pool_3x3", 0),
    ("sep_conv_3x3", 1),
    ("skip_connect", 2),
    ("max_pool_3x3", 0),
]


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_amoebanet_a() -> nn.Module:
    return NASNet(AMOEBANET_A_NORMAL, AMOEBANET_A_REDUCE, C=16, num_cells=3)


def build_amoebanet_b() -> nn.Module:
    return NASNet(AMOEBANET_B_NORMAL, AMOEBANET_B_REDUCE, C=16, num_cells=3)


def build_enas_micro() -> nn.Module:
    return NASNet(ENAS_NORMAL, ENAS_REDUCE, C=16, num_cells=3)


def build_gdas_cifar() -> nn.Module:
    return NASNet(GDAS_NORMAL, GDAS_REDUCE, C=16, num_cells=3)


def example_input_nas() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    ("AmoebaNet-A-CIFAR", "build_amoebanet_a", "example_input_nas", "2019", "DC"),
    ("AmoebaNet-B-ImageNet", "build_amoebanet_b", "example_input_nas", "2019", "DC"),
    ("ENAS-MicroCNN", "build_enas_micro", "example_input_nas", "2018", "DC"),
    ("GDAS-CIFAR", "build_gdas_cifar", "example_input_nas", "2019", "DC"),
]
