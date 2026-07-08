# FAITHFUL PORT of gjoni/trRosetta @ master (original framework: TensorFlow 1.x;
# transcribed from the real repo's own network definition -- NOT a from-scratch
# reimplementation from a paper description)
#
# trRosetta (Yang, Anishchenko, Park, Peng, Ovchinnikov, Baker. 2020, "Improved protein
# structure prediction using predicted inter-residue orientations", PNAS) -- the
# co-evolution-derived-MSA-feature 2D dilated-residual-convolutional network that predicts
# inter-residue distance and orientation (theta/phi/omega) distograms/anglegrams.
# https://github.com/gjoni/trRosetta
#
# Rung-3 justification: the real network is TF1.x (`tf.placeholder`, `tf.contrib.layers`,
# `tf.layers.conv2d`, session-based checkpoint restore) -- TF1.x is not installed and is
# not reasonably installable alongside the rest of the menagerie base env (TF2/Keras3 is
# what's available, and `tf.contrib` was removed entirely in TF2). Transcribed verbatim
# from the real network definition:
#   https://raw.githubusercontent.com/gjoni/trRosetta/master/network/predict.py
#     (network graph: input feature concat, 2D conv trunk (61 residual-dilated-conv
#      blocks, filter width 3, dilation cycling 1->2->4->8->16->1...), instance norm, ELU,
#      output heads for distance/theta/phi/omega/beta-strand-pairing logits+softmax)
#   https://raw.githubusercontent.com/gjoni/trRosetta/master/network/utils.py
#     (feature-prep helpers `reweight`/`msa2pssm`/`fast_dca`; kept for header documentation
#      of the real input feature construction, but NOT executed here -- see note below)
#
# What is kept (every mechanism in the real 2D network, transcribed unmodified): the exact
# trunk topology -- 1x1 conv projecting the 526-channel input feature (`442 + 2*42`) to
# `n2d_filters=64`, InstanceNorm, ELU, then `n2d_layers=61` residual blocks each consisting
# of [conv2d(64, 3x3, dilation=d) -> InstanceNorm -> ELU -> Dropout(0.15) -> conv2d(64,
# 3x3, dilation=d) -> InstanceNorm -> residual-add(the block's own pre-activation input,
# 7 tensors back in the original TF op list) -> ELU], with dilation cycling
# 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, ... (reset to 1 whenever it would exceed 16) -- followed
# by the four output heads: theta-anglegram conv2d(25) + softmax, phi-anglegram conv2d(13)
# + softmax, a symmetrization step (`0.5 * (x + x^T)` over the two spatial axes) applied to
# the trunk features before the *remaining* heads, distance-distogram conv2d(37) +
# softmax, beta-strand-pairing conv2d(3) + softmax (unused downstream in the original
# pipeline but kept -- it is part of the real trained network, not "unused code" we get
# to drop), omega-anglegram conv2d(25) + softmax. All conv2d use `padding='SAME'` (kept
# via PyTorch `padding=dilation` for the 3x3 dilated convs, matching TF SAME-padding
# output size), and InstanceNorm uses `affine=True` (the real network's per-channel
# learnable `beta`/`gamma`, matching `tf.contrib.layers.instance_norm`'s default
# `center=True, scale=True`).
#
# What is dropped/adapted (data preprocessing, not part of the trainable conv-net
# architecture): the real `predict.py` builds `f2d` from a raw multiple-sequence-alignment
# (MSA) via `msa2pssm`/`fast_dca`/`reweight` (`utils.py`) -- one-hot encoding, positional
# weighting, and a 442-channel direct-coupling-analysis (DCA) covariance-inversion
# feature. That feature-extraction math has zero learnable parameters and produces a
# fixed-shape `(1, L, L, 526)` tensor that is purely the network's INPUT, not part of the
# conv trunk; this module accepts that already-built `(1, L, L, 526)` feature tensor
# directly as `example_input_trrosetta()`'s synthetic input (random values, real shape),
# so the actual trainable network -- everything from the first 1x1 conv onward -- runs
# unmodified. TF's `tf.Session`/`tf.train.Saver` checkpoint-averaging-over-5-models loop in
# `predict.py`'s `__main__` is inference/ensembling infrastructure, not architecture, and
# is dropped; this module builds one (randomly initialized) instance of the network.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class ResidualDilatedBlock2d(nn.Module):
    """One residual block from the trRosetta 2D trunk:
    conv2d -> instnorm -> elu -> dropout -> conv2d -> instnorm -> (+residual) -> elu.

    Faithful port of the per-iteration body of the `for _ in range(n2d_layers)` loop in
    the real `network/predict.py` (the residual add is `layers2d[-1] + layers2d[-7]`,
    i.e. adding back the block's pre-activation input from 7 ops earlier in the original
    flat TF op list -- equivalent here to adding the block's own `residual` input).
    """

    def __init__(self, n_filters: int, window: int, dilation: int, dropout_rate: float = 0.15):
        super().__init__()
        pad = dilation * (window - 1) // 2
        self.conv1 = nn.Conv2d(n_filters, n_filters, window, padding=pad, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(n_filters, affine=True)
        self.act1 = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(n_filters, n_filters, window, padding=pad, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(n_filters, affine=True)
        self.act2 = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out + residual)
        return out


class TrRosettaNetwork(nn.Module):
    """Faithful port of the real trRosetta 2D residual-dilated-conv network
    (gjoni/trRosetta network/predict.py), operating on a pre-built (1, L, L, 526) MSA
    feature tensor (see module header for what "pre-built" replaces)."""

    def __init__(
        self,
        in_channels: int = 526,
        n2d_filters: int = 64,
        n2d_layers: int = 61,
        window2d: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, n2d_filters, 1)
        self.stem_norm = nn.InstanceNorm2d(n2d_filters, affine=True)
        self.stem_act = nn.ELU()

        dilation = 1
        blocks = []
        for _ in range(n2d_layers):
            blocks.append(ResidualDilatedBlock2d(n2d_filters, window2d, dilation, dropout_rate))
            dilation *= 2
            if dilation > 16:
                dilation = 1
        self.blocks = nn.ModuleList(blocks)

        # anglegram heads computed BEFORE symmetrization (matches real op order)
        self.to_theta = nn.Conv2d(n2d_filters, 25, 1)
        self.to_phi = nn.Conv2d(n2d_filters, 13, 1)

        # heads computed AFTER symmetrization
        self.to_dist = nn.Conv2d(n2d_filters, 37, 1)
        self.to_bb = nn.Conv2d(n2d_filters, 3, 1)
        self.to_omega = nn.Conv2d(n2d_filters, 25, 1)

    def forward(self, f2d: torch.Tensor) -> dict[str, torch.Tensor]:
        # f2d: (1, L, L, in_channels) -> NCHW for conv2d
        x = f2d.permute(0, 3, 1, 2)
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = self.stem_act(x)

        for block in self.blocks:
            x = block(x)

        logits_theta = self.to_theta(x)
        prob_theta = logits_theta.softmax(dim=1)

        logits_phi = self.to_phi(x)
        prob_phi = logits_phi.softmax(dim=1)

        # symmetrize: 0.5 * (x + x^T) over the two spatial axes (H, W)
        x_sym = 0.5 * (x + x.transpose(2, 3))

        logits_dist = self.to_dist(x_sym)
        prob_dist = logits_dist.softmax(dim=1)

        logits_bb = self.to_bb(x_sym)
        prob_bb = logits_bb.softmax(dim=1)

        logits_omega = self.to_omega(x_sym)
        prob_omega = logits_omega.softmax(dim=1)

        return {
            "dist": prob_dist,
            "theta": prob_theta,
            "phi": prob_phi,
            "omega": prob_omega,
            "bb": prob_bb,
        }


def build_trrosetta():
    return TrRosettaNetwork()


def example_input_trrosetta():
    # Real f2d shape: (1, L, L, 442 + 2*42) = (1, L, L, 526); tiny L for a fast trace.
    L = 12
    return torch.randn(1, L, L, 526)


MENAGERIE_ENTRIES = [
    ("trRosetta", "build_trrosetta", "example_input_trrosetta", 2020, "ported-pytorch"),
]
