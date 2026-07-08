# FAITHFUL PORT of cheminfIBB/pafnucy @ master (original framework: TensorFlow 1.x)
#
# Ported from the model-construction code in the vendored `tfbio/net.py` module
# (function `make_SB_network`, https://gitlab.com/cheminfIBB/pafnucy/-/raw/master/training.py
# imports `tfbio.net`; the `tfbio` package is not published on PyPI/GitHub as a standalone
# distribution, so this port uses the copy embedded in a downstream fork,
# https://github.com/guydurant/Pafnucy/blob/main/net.py, which is byte-for-byte the same
# graph-construction logic Pafnucy's own `training.py`/`predict.py` call).
#
# Pafnucy (Stepniewska-Dziubinska et al., "Development and evaluation of a deep learning
# model for protein-ligand binding affinity prediction", Bioinformatics 2018) predicts
# protein-ligand binding affinity from a voxelized 3D atomic-density grid. Architecture:
# a stack of 3D-conv + ReLU + 3D-maxpool blocks (channels [64, 128, 256], 5x5x5 kernels,
# 2x2x2 "SAME"-padded max pooling) over a (isize, isize, isize, in_chnls) voxel grid,
# flattened and fed through a fully-connected block (sizes [1000, 500, 200]) with dropout,
# ending in a single linear+ReLU output neuron (predicted pK affinity).
#
# TensorFlow 1.x graph-mode (`tf.get_variable`, `tf.variable_scope`, `tf.placeholder`) has
# no path to run in this repo's base torch environment, and TF1.x cannot reasonably be
# installed alongside the pinned torch/torchvision stack -- hence a faithful torch port
# (rung 3) rather than vendoring the original TF1 graph code (rung 2).
#
# "SAME" padding in TF for an odd kernel size k with stride 1 is equivalent to torch's
# `padding=k//2`; TF's SAME max-pool with pool_patch=2, stride=2 on an odd spatial size
# pads asymmetrically (ceil-mode). This port uses `ceil_mode=True` maxpool to reproduce
# that ceil((size)/2) output-size behavior for arbitrary isize (matches `hfsize` calc in
# the original `make_SB_network`: `hfsize = ceil(hfsize / pool_patch)` per conv layer).

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class Hidden3DConvBlock(nn.Module):
    """Faithful port of tfbio.net.hidden_conv3D: conv3d(SAME) -> relu -> maxpool3d(SAME, ceil)."""

    def __init__(self, in_channels, out_channels, conv_patch=5, pool_patch=2):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=conv_patch, padding=conv_patch // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=pool_patch, stride=pool_patch, ceil_mode=True)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class HiddenFCLayer(nn.Module):
    """Faithful port of tfbio.net.hidden_fcl: linear -> relu -> dropout."""

    def __init__(self, in_size, out_size, keep_prob=1.0):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=1.0 - keep_prob)

    def forward(self, x):
        return self.dropout(self.relu(self.linear(x)))


class PafnucySBNetwork(nn.Module):
    """Faithful port of tfbio.net.make_SB_network (Pafnucy's binding-affinity predictor).

    Original: 3D-conv/maxpool stack (conv_channels) -> flatten -> FC stack (dense_sizes)
    with dropout -> single linear+ReLU output neuron predicting binding affinity.
    """

    def __init__(
        self,
        isize=20,
        in_chnls=19,
        osize=1,
        conv_patch=5,
        pool_patch=2,
        conv_channels=(64, 128, 256),
        dense_sizes=(1000, 500, 200),
        keep_prob=1.0,
    ):
        super().__init__()
        self.isize = isize
        self.pool_patch = pool_patch

        conv_blocks = []
        prev_channels = in_chnls
        for out_channels in conv_channels:
            conv_blocks.append(
                Hidden3DConvBlock(prev_channels, out_channels, conv_patch, pool_patch)
            )
            prev_channels = out_channels
        self.conv_blocks = nn.ModuleList(conv_blocks)

        hfsize = isize
        for _ in conv_channels:
            hfsize = -(-hfsize // pool_patch)  # ceil division, matches TF SAME pooling
        self.flat_size = conv_channels[-1] * hfsize**3

        fc_blocks = []
        prev_size = self.flat_size
        for hsize in dense_sizes:
            fc_blocks.append(HiddenFCLayer(prev_size, hsize, keep_prob=keep_prob))
            prev_size = hsize
        self.fc_blocks = nn.ModuleList(fc_blocks)

        self.output_linear = nn.Linear(dense_sizes[-1], osize)
        self.output_relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_chnls, isize, isize, isize) -- torch conv3d channel-first layout
        # (original TF placeholder is channel-last: (batch, isize, isize, isize, in_chnls))
        h = x
        for block in self.conv_blocks:
            h = block(h)
        h_flat = h.reshape(h.shape[0], -1)
        for fc in self.fc_blocks:
            h_flat = fc(h_flat)
        y = self.output_relu(self.output_linear(h_flat))
        return y


def build_pafnucy():
    torch.manual_seed(0)
    return PafnucySBNetwork(isize=8, in_chnls=19, osize=1)


def example_input_pafnucy():
    torch.manual_seed(0)
    return torch.randn(1, 19, 8, 8, 8)


MENAGERIE_ENTRIES = [
    ("Pafnucy", build_pafnucy, example_input_pafnucy, 2018, MENAGERIE_ZOO),
]
