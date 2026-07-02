# FAITHFUL PORT of https://github.com/calico/scBasset @ main (original framework: TensorFlow/Keras)
#
# scBasset (Yuan & Kelley, Nat. Methods 2022): a Basenji-style 1D-CNN that
# maps a one-hot DNA sequence to a per-cell chromatin-accessibility
# probability vector, for sequence-based modeling of single-cell ATAC-seq.
#
# The real architecture lives in scbasset/basenji_utils.py (conv_block,
# conv_tower, dense_block, final, GELU, StochasticReverseComplement,
# StochasticShift, SwitchReverse) and is assembled by
# scbasset/utils.py::make_model(...). It is TensorFlow/Keras (functional
# API), not a base-lib-installable framework here. This module TRANSCRIBES
# the real make_model(...) graph -- same conv tower geometry, same
# GELU/sigmoid activations, same flatten/dense/final head -- into
# self-contained inference-mode PyTorch:
#
#   conv_block()    -> ScBassetConvBlock   (Conv1d + GELU/ReLU + BatchNorm1d, "same" padding)
#   conv_tower()    -> repeat of ScBassetConvBlock with a geometric filter schedule
#   dense_block()   -> ScBassetDenseBlock  (flatten + Linear + BatchNorm1d + GELU)
#   final()         -> ScBassetFinal       (Linear + activation)
#   GELU            -> ScBassetGELU        (the repo's own sigmoid-gated approx,
#                                            not torch's nn.GELU)
#   make_model(...) -> ScBasset nn.Module
#
# StochasticReverseComplement / StochasticShift are training-time-only
# augmentation layers (each returns its input unchanged whenever
# `training` is falsy in the original -- see the `if training:` / `else:
# return seq_1hot` branches) and are therefore the identity in this
# inference-mode port; SwitchReverse similarly reduces to the identity
# once the (always-False at inference) `reverse_bool` flag collapses. Both
# omissions are exact matches of the original's own eval-mode control
# flow, not simplifications of the trained architecture.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ScBassetGELU(nn.Module):
    """Port of basenji_utils.py::GELU -- NOT torch.nn.GELU; the repo uses its
    own sigmoid(1.702 * x) * x approximation."""

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class ScBassetConvBlock(nn.Module):
    """Port of basenji_utils.py::conv_block (activation='gelu', batch_norm=True,
    residual=False, conv_type='standard', 'same' padding path)."""

    def __init__(self, in_channels, filters, kernel_size=1, pool_size=1):
        super().__init__()
        self.act = ScBassetGELU()
        pad = kernel_size // 2  # 'same' padding for odd kernel_size / stride 1
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(
            filters, momentum=0.10, eps=1e-3
        )  # tf momentum=0.90 -> torch momentum=1-0.90
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else None

    def forward(self, x):
        # x: B x C x L
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class ScBassetDenseBlock(nn.Module):
    """Port of basenji_utils.py::dense_block (flatten=True, batch_norm=True,
    activation='gelu' applied by the caller AFTER this block, matching
    make_model's ``current = dense_block(...); current = GELU()(current)``)."""

    def __init__(self, in_features, units):
        super().__init__()
        self.linear = nn.Linear(in_features, units, bias=False)
        self.bn = nn.BatchNorm1d(units, momentum=0.10, eps=1e-3)

    def forward(self, x_flat):
        x = self.linear(x_flat)
        x = self.bn(x)
        return x


class ScBassetFinal(nn.Module):
    """Port of basenji_utils.py::final (activation='sigmoid', flatten=False --
    input is already B x features here since make_model applies it after
    the flattened dense_block)."""

    def __init__(self, in_features, units):
        super().__init__()
        self.linear = nn.Linear(in_features, units, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class ScBasset(nn.Module):
    """Port of scbasset/utils.py::make_model(bottleneck_size, n_cells, seq_len).

    Faithfully mirrors the real conv geometry: 1 initial conv_block
    (filters=288, kernel=17, pool=3), a 6-repeat conv_tower with a
    filters_mult=1.122 geometric schedule (kernel=5, pool=2 each), a 1x1
    conv_block (filters=256), a flatten+dense bottleneck (dropout dropped
    at inference, matching eval-mode nn.Dropout), a GELU, and a final
    sigmoid dense head over n_cells. StochasticReverseComplement /
    StochasticShift / SwitchReverse are the identity at inference (see
    module docstring)."""

    def __init__(self, bottleneck_size=32, n_cells=8, seq_len=1344):
        super().__init__()
        self.seq_len = seq_len

        self.stem = ScBassetConvBlock(4, 288, kernel_size=17, pool_size=3)

        repeat = 6
        filters_init = 288
        filters_mult = 1.122
        tower_blocks = []
        in_ch = filters_init
        rep_filters = float(filters_init)
        for _ in range(repeat):
            out_ch = int(round(rep_filters))
            tower_blocks.append(ScBassetConvBlock(in_ch, out_ch, kernel_size=5, pool_size=2))
            in_ch = out_ch
            rep_filters *= filters_mult
        self.tower = nn.ModuleList(tower_blocks)

        self.post_conv = ScBassetConvBlock(in_ch, 256, kernel_size=1, pool_size=1)

        post_len = seq_len // 3
        for _ in range(repeat):
            post_len = math.ceil(post_len / 2)
        self.flat_features = 256 * post_len

        self.dense = ScBassetDenseBlock(self.flat_features, bottleneck_size)
        self.dense_act = ScBassetGELU()
        self.final = ScBassetFinal(bottleneck_size, n_cells)

    def forward(self, seq_1hot):
        # seq_1hot: B x seq_len x 4 (one-hot DNA), matching the repo's Keras
        # Input(shape=(seq_len, 4)) convention.
        x = seq_1hot.transpose(1, 2)  # -> B x 4 x seq_len for Conv1d
        x = self.stem(x)
        for block in self.tower:
            x = block(x)
        x = self.post_conv(x)
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1
        )  # flatten positional axis (seq-major, like tf Reshape)
        x = self.dense(x)
        x = self.dense_act(x)
        x = self.final(x)
        return x


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_SEQ_LEN = 192
_N_CELLS = 8
_BOTTLENECK = 16


def build_scbasset():
    torch.manual_seed(0)
    model = ScBasset(bottleneck_size=_BOTTLENECK, n_cells=_N_CELLS, seq_len=_SEQ_LEN)
    model.eval()
    return model


def example_input_scbasset():
    torch.manual_seed(0)
    idx = torch.randint(0, 4, (1, _SEQ_LEN))
    seq_1hot = torch.nn.functional.one_hot(idx, num_classes=4).float()
    return seq_1hot


MENAGERIE_ENTRIES = [
    ("scBasset", "build_scbasset", "example_input_scbasset", 2022, MENAGERIE_ZOO),
]
