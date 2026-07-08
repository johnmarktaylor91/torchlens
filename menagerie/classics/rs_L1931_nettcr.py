# FAITHFUL PORT of mnielLab/NetTCR-2.0 @ main (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/mnielLab/NetTCR-2.0/main/nettcr_architectures.py
#
# Montemurro, Schuster, Povlsen, Bentzen, Jurtz, Chronister, Crinklaw, Hadrup,
# Winther, Sette, Peters, Nielsen, 2021 (Communications Biology) "NetTCR-2.0
# enables accurate prediction of TCR-peptide binding by using paired TCRalpha and
# TCRbeta sequence data". NetTCR-2.0's `nettcr_ab` is the paper's full model: three
# parallel input towers (peptide, TCR CDR3-alpha, TCR CDR3-beta), each processed
# by five parallel 1D convolutions of kernel sizes {1, 3, 5, 7, 9} (16 filters
# each, sigmoid activation, 'same' padding) followed by global max pooling over
# the sequence axis, concatenated per-tower, then all three towers' pooled
# features are concatenated and fed through a 32-unit dense layer into a single
# sigmoid binding-probability output. The real repo (`nettcr_architectures.py`)
# is Keras/TensorFlow (`from keras.layers import ...`), with no PyTorch code
# anywhere in the repo or its branches -- per the ladder this is a rung-3 case
# (framework that can't run in the base torch env), so the architecture below is
# a faithful line-for-line transcription of the real `nettcr_ab()` Keras
# functional-API graph into self-contained torch: every tower, every
# kernel-size branch, the sigmoid activations, and the pooling/concatenation
# topology are reproduced exactly (only the mechanical Keras->torch API
# translation -- e.g. `Conv1D(..., padding='same')` -> `nn.Conv1d` with
# kernel_size-derived symmetric padding, since kernel sizes here are all odd, and
# `GlobalMaxPooling1D()` -> `torch.amax` over the sequence axis after
# transposing Keras's channels-last `(batch, seq, chan)` layout to torch's
# channels-first `(batch, chan, seq)`).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

_KERNEL_SIZES = (1, 3, 5, 7, 9)


class _MultiKernelTower(nn.Module):
    """Faithful port of one NetTCR input tower: five parallel Conv1D(16,
    kernel_size=k, padding='same', activation='sigmoid') branches, each
    followed by GlobalMaxPooling1D, concatenated along the channel axis.
    Matches the real repo's per-tower construction in `nettcr_ab`/
    `nettcr_one_chain` (identical for the peptide, CDR3-alpha, and CDR3-beta
    towers -- only weights, not architecture, differ between towers)."""

    def __init__(self, in_channels, n_filters=16, kernel_sizes=_KERNEL_SIZES):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels, n_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, in_channels, seq_len] (torch channels-first; the real
        # Keras model consumes channels-last [batch, seq_len, in_channels])
        pooled = []
        for conv in self.convs:
            out = self.activation(conv(x))
            pooled.append(torch.amax(out, dim=2))  # GlobalMaxPooling1D
        return torch.cat(pooled, dim=1)


class NetTCR_AB(nn.Module):
    """Faithful port of the real repo's `nettcr_ab()`: three `_MultiKernelTower`
    branches (peptide, CDR3-alpha, CDR3-beta) feeding a 32-unit sigmoid dense
    layer and a final sigmoid binding-probability output. `in_channels=20`
    matches the real BLOSUM50 20-amino-acid encoding used by `utils.py`."""

    def __init__(self, in_channels=20, n_filters=16, dense_units=32):
        super().__init__()
        self.pep_tower = _MultiKernelTower(in_channels, n_filters)
        self.cdr3a_tower = _MultiKernelTower(in_channels, n_filters)
        self.cdr3b_tower = _MultiKernelTower(in_channels, n_filters)

        tower_out_dim = n_filters * len(_KERNEL_SIZES)
        self.dense = nn.Linear(tower_out_dim * 3, dense_units)
        self.dense_activation = nn.Sigmoid()
        self.out = nn.Linear(dense_units, 1)
        self.out_activation = nn.Sigmoid()

    def forward(self, pep, cdr3a, cdr3b):
        pep_feat = self.pep_tower(pep)
        cdr3a_feat = self.cdr3a_tower(cdr3a)
        cdr3b_feat = self.cdr3b_tower(cdr3b)

        cat = torch.cat([pep_feat, cdr3a_feat, cdr3b_feat], dim=1)
        dense = self.dense_activation(self.dense(cat))
        out = self.out_activation(self.out(dense))
        return out


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_nettcr_ab():
    model = NetTCR_AB(in_channels=20, n_filters=16, dense_units=32)
    model.eval()
    return model


def example_input_nettcr_ab():
    torch.manual_seed(0)
    batch = 1
    # real repo's fixed sequence lengths: peptide=9, CDR3=30 (BLOSUM50 20-aa
    # encoding -> 20 channels); torch channels-first [batch, 20, seq_len].
    pep = torch.randn(batch, 20, 9)
    cdr3a = torch.randn(batch, 20, 30)
    cdr3b = torch.randn(batch, 20, 30)
    return (pep, cdr3a, cdr3b)


MENAGERIE_ENTRIES = [
    ("NetTCR-2.0", build_nettcr_ab, example_input_nettcr_ab, 2021, "ported-pytorch"),
]
