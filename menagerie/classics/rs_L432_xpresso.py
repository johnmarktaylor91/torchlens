# FAITHFUL PORT of vagarwal87/Xpresso @ master (original framework: Keras 2.x / TensorFlow)
# https://raw.githubusercontent.com/vagarwal87/Xpresso/master/Fig1_S2/Xpresso.py
#
# Agarwal, Shendure 2020 (Cell Reports) "Predicting mRNA Abundance Directly
# from Genomic Sequence Using Deep Convolutional Neural Networks" -- Xpresso,
# a CNN that predicts steady-state mRNA expression directly from promoter
# sequence plus a small set of mRNA half-life-related features. The real
# model is built in `objective()` (Fig1_S2/Xpresso.py) as a Keras functional-
# API graph: a one-hot promoter-sequence window (`Input(shape=(rightpos -
# leftpos, 4))`) goes through 1-4 `Conv1D(..., padding="same", dilation_rate=
# ..., kernel_initializer="glorot_normal")` + `MaxPooling1D` stages (the exact
# depth/filter counts/kernel widths/dilation rates/pool sizes are all
# hyperparameter-search fields -- `numconvlayers` is a nested "one"/"two"/
# "three"/"four" choice tree), then `Flatten()`, then `Concatenate()` with an
# 8-dimensional half-life-feature vector (`Input(shape=(8,), name="halflife")`
# -- `halflifedata = table[:, 1:9]`, `setup_training_files.py`), then 1-2
# `Dense` + activation + `Dropout` stages (`numdenselayers` is likewise a
# "one"/"two" hyperparameter choice), ending in a scalar `Dense(1)` regression
# output (log10 mRNA abundance). TensorFlow/Keras is not an installed base
# lib and installing the full TF stack for one small architecture is not
# reasonable, so per rung 3 this is transcribed faithfully into self-
# contained base-env torch: every real layer/mechanism (dilated same-padded
# 1D convs, max-pools, concat-fusion with the half-life vector, dense+
# activation+dropout stack, scalar regression head) is preserved, using the
# real repo's own "best manually identified" hyperparameter set (from
# `main()`'s `--bestmanual` branch, the paper's reported best config):
# leftpos=8500, rightpos=11500 (3000bp promoter window), activationFxn="relu",
# numFiltersConv1=64, filterLenConv1=5, dilRate1=1, maxPool1=10,
# numFiltersConv2=64, filterLenConv2=5, dilRate2=1, maxPool2=20 (2 conv
# layers -- "numconvlayers1"=="two"), dense1=100, dropout1=0.5, one dense
# layer ("numdenselayers"=="one"). `padding="same"` (Keras) is realized in
# torch as an explicit symmetric pad sized from kernel/dilation (torch's
# `Conv1d` has no native `padding="same"` + `dilation` combination on the
# stable Conv1d op used identically across torch versions, so the padding
# amount is computed explicitly here, matching the same output length Keras
# produces for odd kernel sizes and dilation_rate=1).

import torch
import torch.nn as nn


class SamePadConv1d(nn.Module):
    """Conv1d with Keras-style `padding="same"` semantics (stride=1).

    The real model uses `Conv1D(..., padding='same', dilation_rate=...)` --
    Keras computes symmetric (or near-symmetric, extra pad on the right for
    even totals) zero-padding so the output length equals the input length.
    This mirrors that behavior exactly for stride=1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        effective_kernel = (kernel_size - 1) * dilation + 1
        pad_total = effective_kernel - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        self.pad = nn.ConstantPad1d((pad_left, pad_right), 0.0)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        nn.init.xavier_normal_(self.conv.weight)  # Keras "glorot_normal"
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class Xpresso(nn.Module):
    """Faithful port of the real Keras `objective()` model graph.

    Real "best manually identified" hyperparameters (Fig1_S2/Xpresso.py
    `main()` `--bestmanual` branch): 2 conv+pool stages, dense1=100,
    dropout1=0.5, 1 dense stage, ReLU activations throughout.
    """

    def __init__(
        self,
        seq_len=3000,  # rightpos(11500) - leftpos(8500)
        n_halflife_feat=8,
        num_filters_conv1=64,
        filter_len_conv1=5,
        dil_rate1=1,
        max_pool1=10,
        num_filters_conv2=64,
        filter_len_conv2=5,
        dil_rate2=1,
        max_pool2=20,
        dense1=100,
        dropout1=0.5,
    ):
        super().__init__()
        self.conv1 = SamePadConv1d(4, num_filters_conv1, filter_len_conv1, dilation=dil_rate1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(max_pool1)

        self.conv2 = SamePadConv1d(
            num_filters_conv1, num_filters_conv2, filter_len_conv2, dilation=dil_rate2
        )
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(max_pool2)

        conv_out_len = seq_len // max_pool1 // max_pool2
        flat_dim = conv_out_len * num_filters_conv2

        self.dense1 = nn.Linear(flat_dim + n_halflife_feat, dense1)
        self.act_dense1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)

        self.output_layer = nn.Linear(dense1, 1)

    def forward(self, input_promoter, halflife):
        # input_promoter: (N, seq_len, 4) one-hot -- Keras Conv1D channel-last;
        # torch Conv1d is channel-first, so transpose to (N, 4, seq_len).
        x = input_promoter.transpose(1, 2)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = torch.cat([x, halflife], dim=1)

        x = self.dense1(x)
        x = self.act_dense1(x)
        x = self.dropout1(x)

        return self.output_layer(x)


def build_xpresso():
    # Real best-manual hyperparameters kept as-is except seq_len (real
    # rightpos-leftpos=3000 is kept exactly -- shrinking it would change the
    # real conv-output-length arithmetic the model relies on).
    return Xpresso()


def example_input_xpresso():
    n = 2
    seq_len = 3000
    n_halflife_feat = 8
    # One-hot promoter window: for each position, exactly one of 4 channels is 1.
    idx = torch.randint(0, 4, (n, seq_len))
    input_promoter = torch.nn.functional.one_hot(idx, num_classes=4).float()
    halflife = torch.rand(n, n_halflife_feat)
    return (input_promoter, halflife)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("Xpresso", "build_xpresso", "example_input_xpresso", 2020, "ported-pytorch"),
]
