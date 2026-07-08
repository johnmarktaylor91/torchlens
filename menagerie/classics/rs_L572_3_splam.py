# SOURCE: vendored from Kuanhao-Chao/splam @ main
# Original: src/splam/splam.py (ResidualUnit, Skip, SPLAM)
"""SPLAM: a SpliceAI-style dilated residual CNN for splice-site prediction.

SPLAM predicts per-base splice-donor / splice-acceptor / neither
probabilities from a one-hot-encoded DNA window (4 channels: A/C/G/T),
via a stack of grouped, dilated Conv1d "residual units" (increasing kernel
width and dilation rate in stages) with periodic 1x1 "skip" convolutions
accumulating a running skip-connection, followed by a final 1x1 conv +
softmax classification head over 3 classes (donor / acceptor / neither).
Splam is trained on splice junctions rather than single splice sites
(distinguishing it from its SpliceAI-lineage predecessors) but shares the
same residual-dilated-CNN backbone family.

Reference: https://github.com/Kuanhao-Chao/splam
"""

import numpy as np
import torch
from torch.nn import BatchNorm1d, Conv1d, LeakyReLU, Module, ModuleList, Softmax

MENAGERIE_ZOO = "vendored-pytorch"

CARDINALITY_ITEM = 16


# --- vendored from src/splam/splam.py ---


class ResidualUnit(Module):
    def __init__(self, l, w, ar, bot_mul=1):  # noqa: E741 (vendored verbatim)
        super().__init__()
        bot_channels = int(round(l * bot_mul))
        self.batchnorm1 = BatchNorm1d(l)
        self.relu = LeakyReLU(0.1)
        self.batchnorm2 = BatchNorm1d(l)
        self.C = bot_channels // CARDINALITY_ITEM
        self.conv1 = Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2, groups=self.C)
        self.conv2 = Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2, groups=self.C)

    def forward(self, x, y):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x1)))
        return x + x2, y


class Skip(Module):
    def __init__(self, l):  # noqa: E741 (vendored verbatim)
        super().__init__()
        self.conv = Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class SPLAM(Module):
    def __init__(
        self,
        L=64,
        W=np.array([11] * 8 + [21] * 4 + [41] * 4),
        AR=np.array([1] * 4 + [4] * 4 + [10] * 4 + [25] * 4),
    ):
        super().__init__()
        self.CL = 2 * (AR * (W - 1)).sum()  # context length
        self.conv1 = Conv1d(4, L, 1)
        self.skip1 = Skip(L)
        self.residual_blocks = ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_blocks.append(ResidualUnit(L, w, r))
            if (i + 1) % 4 == 0:
                self.residual_blocks.append(Skip(L))
        if (len(W) + 1) % 4 != 0:
            self.residual_blocks.append(Skip(L))
        self.last_cov = Conv1d(L, 3, 1)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x, skip = self.skip1(self.conv1(x), 0)
        for m in self.residual_blocks:
            x, skip = m(x, skip)
        # predicting pb for every bp
        return self.softmax(self.last_cov(skip))


# --- staging: tiny-size builder + example input ---

# L must stay a multiple of CARDINALITY_ITEM (16) so bot_channels//16 >= 1
# (groups must be a positive integer); the real default is L=64. W/AR are
# shrunk from the real 16-stage schedule ([11]*8+[21]*4+[41]*4 /
# [1]*4+[4]*4+[10]*4+[25]*4) to a single 4-stage block, preserving the same
# per-stage (kernel_width, dilation_rate) pairing pattern.
_L = 16
_W = np.array([11] * 4)
_AR = np.array([1] * 4)
_SEQ_LEN = 100


def build_splam():
    model = SPLAM(L=_L, W=_W, AR=_AR)
    model.eval()
    return model


def example_input_splam():
    return torch.rand(2, 4, _SEQ_LEN)


MENAGERIE_ENTRIES = [
    (
        "Splam",
        "build_splam",
        "example_input_splam",
        2023,
        "vendored-pytorch",
    ),
]
