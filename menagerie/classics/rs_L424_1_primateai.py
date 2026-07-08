# FAITHFUL PORT of Illumina/PrimateAI @ master (original framework: Keras 1.x / TF1.x)
# https://raw.githubusercontent.com/Illumina/PrimateAI/master/source/model.py
#
# Sundaram et al. 2018 (Nat. Genet.) "Predicting the clinical impact of human mutation
# with deep neural networks" -- PrimateAI v1's `primateAI_model()` (Code/model.py) is a
# dilated-residual 1D-conv network that scores a missense variant by jointly encoding
# the wild-type and mutant 51-residue windows together with three conservation tracks
# (primate / mammal / other-vertebrate PSSM-style one-hot columns, each length-51 x 20).
#
# Faithful structure transcribed from the real `model.py` (Keras `Model` functional API,
# TF1.x-era `keras.layers.convolutional.Conv1D` / `keras.layers.normalization
# .BatchNormalization` / `merge`):
#   1. Two small auxiliary sub-networks (`get_ss_model`, `get_sa_model`) -- each a
#      6-residual-block, pre-activation (BN->ReLU->Conv1D, x2, +skip) dilated Conv1D
#      stack with L=40 filters, W=[5,5,5] kernel widths (2 blocks per width), a running
#      "denseforskip" 1x1-conv accumulator added into a parallel `skip` path -- run on
#      the ELEMENTWISE SUM of the three conservation tracks (`input2+input3+input4`).
#      These stand in for real pretrained secondary-structure (`ss`) and solvent-
#      accessibility (`sa`) predictors, called as sub-models inside the main graph.
#   2. Five 1x1 convs project (orig_seq, snp_seq, conservation_primates,
#      conservation_mammals, conservation_otherspecies) each to L=40 channels.
#   3. `struc` (ss_model output) and `solv` (sa_model output) are added into BOTH the
#      orig and snp branches together with all three conservation projections
#      (`merge_orig_conserv`, `merge_snp_conserv`).
#   4. Each branch passes through one non-residual pre-activation residual_unit
#      (kernel width W[0]=5, no skip-add -- `residual='no'`), then the two branches are
#      concatenated on the channel axis TWICE (once for the running `conv` path, once
#      for the running `skip` path) and each reduced back to L=40 channels via a
#      kernel-5 "reduce" Conv1D.
#   5. Same 6-residual-block / 3-width dilated stack as the sub-models (with the
#      running skip-accumulator pattern) processes the merged conv/skip pair.
#   6. A final non-dilated residual unit, then a 1x1 Conv1D to a single channel with
#      sigmoid, then `GlobalMaxPooling1D` over the sequence axis -> scalar risk score
#      in [0, 1] per example.
#
# No architectural changes: same residual_unit mechanism (pre-activation BN->ReLU->Conv,
# twice, optional residual add), same filter count/kernel widths/dilation schedule, same
# skip-accumulator wiring, same dual-branch (orig/snp) with shared conservation fusion,
# same final GlobalMaxPooling1D + sigmoid head. `BatchNorm1d` replaces Keras
# `BatchNormalization` (channels-first NCL vs Keras' channels-last NLC -- the port keeps
# torch's native NCL Conv1d layout, so tensors are transposed once at the input boundary
# instead of threading `channels_last` through every layer). `keras.layers.add` /
# `keras.layers.concatenate` become `torch.add` / `torch.cat(dim=1)` (channel axis in
# NCL). This port omits `weight.load_weights(...)` (no public v1 weights bundled in the
# repo) and keeps the real per-layer dimensionality (`L=40`, `N=[2,2,2]`, `W=[5,5,5]`,
# `AR=[1,1,1]`, `seq_length=51`, 20 amino-acid channels) with random init.

import torch
import torch.nn as nn


class ResidualUnit1D(nn.Module):
    """Faithful port of the real `residual_unit(nb_fil, f_len, ar, indx, residual)`.

    Pre-activation: BN->ReLU->Conv1D, twice, with an optional residual add of the
    block's input (Keras `residual='yes'`/`'no'` switch).
    """

    def __init__(self, channels, kernel_size, dilation, residual=True):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.residual = residual

    def forward(self, x):
        h = self.conv1(torch.relu(self.bn1(x)))
        h = self.conv2(torch.relu(self.bn2(h)))
        if self.residual:
            return h + x
        return h


class DilatedResidualStack(nn.Module):
    """Faithful port of the shared `N=[2,2,2]`, `W=[5,5,5]`, `AR=[1,1,1]` residual
    stack + running skip-accumulator pattern used by `get_ss_model`, `get_sa_model`,
    and the main model's post-merge trunk.
    """

    def __init__(self, channels=40, n_blocks=(2, 2, 2), widths=(5, 5, 5), dilations=(1, 1, 1)):
        super().__init__()
        self.stages = nn.ModuleList()
        self.skip_dense = nn.ModuleList()
        for n, w, ar in zip(n_blocks, widths, dilations):
            stage = nn.ModuleList(
                [ResidualUnit1D(channels, w, ar, residual=True) for _ in range(n)]
            )
            self.stages.append(stage)
            self.skip_dense.append(nn.Conv1d(channels, channels, kernel_size=1))
        self.final_unit = ResidualUnit1D(channels, kernel_size=1, dilation=1, residual=True)

    def forward(self, conv, skip):
        for stage, dense in zip(self.stages, self.skip_dense):
            for block in stage:
                conv = block(conv)
            skip = skip + dense(conv)
        return self.final_unit(skip)


class SubModel(nn.Module):
    """Faithful port of `get_ss_model` / `get_sa_model`: parallel `conv`/`skip` 1x1
    projections, down-projected, then the shared dilated residual stack (mirrors the
    real `conv = Conv1D(...)(conv)`; `skip = Conv1D(...)(skip)` down-project pair
    before the residual-block loop).
    """

    def __init__(self, in_channels=20, channels=40):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.down_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.down_skip = nn.Conv1d(channels, channels, kernel_size=1)
        self.stack = DilatedResidualStack(channels)

    def forward(self, x):
        conv = self.down_conv(self.proj(x))
        skip = self.down_skip(self.skip_proj(x))
        return self.stack(conv, skip)


class PrimateAI(nn.Module):
    """Faithful port of `primateAI_model()` (Illumina/PrimateAI `source/model.py`).

    Inputs (torch NCL layout, channels=20 one-hot amino acids, length=51):
      orig_seq, snp_seq, conservation_primates, conservation_mammals,
      conservation_otherspecies -- each `(batch, 20, 51)`.
    Output: `(batch,)` scalar pathogenicity score in [0, 1].
    """

    def __init__(self, seq_length=51, channels=40, in_channels=20):
        super().__init__()
        self.ss_model = SubModel(in_channels, channels)
        self.sa_model = SubModel(in_channels, channels)

        self.conv_orig_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.conv_snp_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.conv_primate_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.conv_mammal_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.conv_vert_proj = nn.Conv1d(in_channels, channels, kernel_size=1)

        self.orig_residual = ResidualUnit1D(channels, kernel_size=5, dilation=1, residual=False)
        self.snp_residual = ResidualUnit1D(channels, kernel_size=5, dilation=1, residual=False)

        self.conv_reduce = nn.Conv1d(2 * channels, channels, kernel_size=5, padding=2)
        self.skip_reduce = nn.Conv1d(2 * channels, channels, kernel_size=5, padding=2)

        self.trunk = DilatedResidualStack(channels)
        self.final_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(
        self,
        orig_seq,
        snp_seq,
        conservation_primates,
        conservation_mammals,
        conservation_otherspecies,
    ):
        cons_sum = conservation_primates + conservation_mammals + conservation_otherspecies
        struc = self.ss_model(cons_sum)
        solv = self.sa_model(cons_sum)

        conv_orig = self.conv_orig_proj(orig_seq)
        conv_snp = self.conv_snp_proj(snp_seq)
        conv_primate = self.conv_primate_proj(conservation_primates)
        conv_mammal = self.conv_mammal_proj(conservation_mammals)
        conv_vert = self.conv_vert_proj(conservation_otherspecies)

        merge_orig = conv_orig + conv_primate + conv_mammal + conv_vert + struc + solv
        merge_snp = conv_snp + conv_primate + conv_mammal + conv_vert + struc + solv

        conv_orig1 = self.orig_residual(merge_orig)
        conv_snp1 = self.snp_residual(merge_snp)

        merged = torch.cat([conv_orig1, conv_snp1], dim=1)
        conv = self.conv_reduce(merged)
        skip = self.skip_reduce(merged)

        trunk_out = self.trunk(conv, skip)
        out = torch.sigmoid(self.final_conv(trunk_out))
        return out.amax(dim=-1).squeeze(-1)


def build_primateai():
    return PrimateAI(seq_length=51, channels=40, in_channels=20)


def example_input_primateai():
    batch, length, channels = 2, 51, 20
    orig_seq = torch.rand(batch, channels, length)
    snp_seq = torch.rand(batch, channels, length)
    cons_primates = torch.rand(batch, channels, length)
    cons_mammals = torch.rand(batch, channels, length)
    cons_other = torch.rand(batch, channels, length)
    # tuple of 5 positional args -- torchlens `input_args` unpacks tuples/lists as
    # multiple positional arguments to `forward()`.
    return (orig_seq, snp_seq, cons_primates, cons_mammals, cons_other)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("PrimateAI", build_primateai, example_input_primateai, 2018, "ported-pytorch"),
]
