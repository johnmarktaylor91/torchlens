# FAITHFUL PORT of younglululu/DeepPINK @ master (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/younglululu/DeepPINK/master/run_withKnockoff_all.py
#
# Lu, Fan, Lv, Noble 2018 (NeurIPS) "DeepPINK: reproducible feature selection in deep neural
# networks". The real `build_DNN(p, coeff)` in `run_withKnockoff_all.py` is Keras/TF
# (`keras.layers.LocallyConnected1D`, `Dense`), a framework not in this project's PyTorch base
# environment, so the architecture is transcribed faithfully into self-contained torch rather
# than vendored. Every layer of the real functional-API graph is reproduced 1:1:
#
#   Input(shape=(p, 2))                                            # [original_j, knockoff_j] per feature j
#   -> LocallyConnected1D(filterNum=1, kernel_size=1, use_bias=True)   # independent per-position Dense(2->1)
#   -> LocallyConnected1D(1,           kernel_size=1, use_bias=True)   # independent per-position Dense(1->1)
#   -> Flatten()                                                    # (batch, p, 1) -> (batch, p)
#   -> Dense(p, activation='relu',   kernel_regularizer=l1(coeff))
#   -> Dense(p, activation='relu',   kernel_regularizer=l1(coeff))
#   -> Dense(1, activation='sigmoid')
#
# `keras.layers.LocallyConnected1D` (unlike `Conv1D`) applies an *unshared*, independently
# weighted Dense transform at every one of the `p` positions along the sequence axis --
# with `kernel_size=1` this is exactly a per-position `Linear(input_dim -> filters)`, one
# distinct weight matrix per position, no weight sharing across positions (the paper's
# "locally connected" layer is what makes each input feature's [original, knockoff] pair
# map to its own scalar without mixing across features). `LocallyConnectedPort` below
# reproduces that per-position-independent-weight semantics directly with a
# `(p, out_ch, in_ch)`-shaped parameter tensor and a batched matmul, matching Keras'
# `LocallyConnected1D(kernel_size=1)` output exactly (up to numerically-irrelevant weight
# initialization). L1 kernel regularization (`kernel_regularizer=l1(coeff)`) is a training-time
# loss term, not part of the forward computation graph, so it has no forward-pass
# counterpart here (as with any weight-decay/regularization term in a ported architecture).

import torch
from torch import nn


class LocallyConnectedPort(nn.Module):
    """Faithful port of keras.layers.LocallyConnected1D(filters, kernel_size=1).

    Applies an independently-weighted Linear(in_channels -> filters) at each of the
    `positions` steps along dim=1 (no weight sharing across positions), matching Keras'
    LocallyConnected1D semantics at kernel_size=1 / stride=1 / padding='valid'.
    """

    def __init__(self, positions: int, in_channels: int, filters: int, use_bias: bool = True):
        super().__init__()
        self.positions = positions
        self.filters = filters
        # one independent (in_channels -> filters) weight per position
        self.weight = nn.Parameter(torch.empty(positions, filters, in_channels))
        nn.init.xavier_normal_(self.weight)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(positions, filters))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (batch, positions, in_channels) -> (batch, positions, filters)
        out = torch.einsum("bpi,pfi->bpf", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class DeepPINK(nn.Module):
    """Faithful port of build_DNN(p, coeff) from run_withKnockoff_all.py."""

    def __init__(self, p: int, filter_num: int = 1):
        super().__init__()
        self.p = p
        self.local1 = LocallyConnectedPort(p, in_channels=2, filters=filter_num, use_bias=True)
        self.local2 = LocallyConnectedPort(p, in_channels=filter_num, filters=1, use_bias=True)
        self.dense1 = nn.Linear(p, p)
        self.dense2 = nn.Linear(p, p)
        self.out = nn.Linear(p, 1)

    def forward(self, x):
        # x: (batch, p, 2) -- [:, :, 0] = original feature, [:, :, 1] = knockoff feature
        x = self.local1(x)  # (batch, p, filter_num)
        x = self.local2(x)  # (batch, p, 1)
        x = x.flatten(start_dim=1)  # (batch, p)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.sigmoid(self.out(x))
        return x


MENAGERIE_ZOO = "ported-pytorch"


def build_deeppink():
    model = DeepPINK(p=32, filter_num=1)
    model.eval()
    return model


def example_input_deeppink():
    batch = 1
    p = 32
    return (torch.randn(batch, p, 2),)


MENAGERIE_ENTRIES = [
    (
        "DeepPINK",
        build_deeppink,
        example_input_deeppink,
        2018,
        MENAGERIE_ZOO,
    ),
]
