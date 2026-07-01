# FAITHFUL PORT of awni/ecg @ master (original framework: Keras/TF)
# https://raw.githubusercontent.com/awni/ecg/master/ecg/network.py
# https://raw.githubusercontent.com/awni/ecg/master/examples/cinc17/config.json
# https://raw.githubusercontent.com/awni/ecg/master/ecg/train.py
#
# Rajpurkar/Hannun et al. 2017 "Cardiologist-Level Arrhythmia Detection with
# Convolutional Neural Networks" (a.k.a. "DeepHeart") -- `ecg/network.py`
# builds the model with pure Keras functional-API calls (no PyTorch); the
# 1D-ResNet architecture is faithfully TRANSCRIBED into torch below,
# reproducing `build_network()`'s real control flow (`add_resnet_layers` /
# `resnet_block` / `add_conv_weight` / `_bn_relu` / `add_output_layer`) with
# the exact hyperparameters shipped in `examples/cinc17/config.json`:
#   conv_subsample_lengths = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]  (16 blocks)
#   conv_filter_length = 16, conv_num_filters_start = 32,
#   conv_activation = "relu", conv_dropout = 0.2, conv_num_skip = 2,
#   conv_increase_channels_at = 4
# and `train.py`'s dataset-derived args: `input_shape=[None, 1]` (raw
# single-lead ECG signal) and `num_categories=4` (the CINC17 challenge's
# Normal/AFib/Other/Noisy classes -- the actual label set this repo's
# `examples/cinc17` config trains against, per the repo README).
#
# Reproduced faithfully:
#   - `add_conv_weight`: Conv1d(kernel=16, padding='same' equivalent,
#     stride=subsample_length, no bias since Keras `Conv1D` default `use_bias
#     =True` but original omits explicit bias arg -- kept as PyTorch default
#     bias=True to match Keras `Conv1D` default) -> ported as-is.
#   - `_bn_relu`: BatchNorm1d -> ReLU -> (Dropout if dropout>0).
#   - `resnet_block`: shortcut = MaxPool1d(subsample_length) on the block
#     input, zero-padded (channel-doubled via zero concat) every
#     `conv_increase_channels_at` blocks (block_index>0); main path applies
#     `conv_num_skip` (=2) conv sub-layers, first BN-ReLU-dropout skipped on
#     the very first sub-layer of the very first block (matches original's
#     `if not (block_index == 0 and i == 0)` guard); output = shortcut + main.
#   - `add_resnet_layers`: initial conv+BN-ReLU, then one `resnet_block` per
#     entry in `conv_subsample_lengths`, final BN-ReLU.
#   - `add_output_layer`: per-timestep Dense(num_categories) + softmax
#     (`TimeDistributed(Dense(...))` -> `nn.Linear` applied over the last
#     dim, since PyTorch `nn.Linear` already broadcasts over leading dims
#     the same way Keras `TimeDistributed(Dense)` does).
# Only mechanical import/framework substitutions (Keras layer classes ->
# torch nn.Module equivalents); no architectural invention.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_1d(x, kernel_size, stride):
    """Reproduce Keras Conv1D(padding='same') for stride>=1: pad so the
    output length is ceil(input_length / stride), matching TF's SAME
    padding convention (asymmetric pad favors the right side)."""
    input_len = x.shape[-1]
    out_len = (input_len + stride - 1) // stride
    pad_needed = max(0, (out_len - 1) * stride + kernel_size - input_len)
    pad_left = pad_needed // 2
    pad_right = pad_needed - pad_left
    return F.pad(x, (pad_left, pad_right))


class ConvSame1d(nn.Module):
    """Conv1d with Keras-style 'same' padding (original: `padding='same'`)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = _same_pad_1d(x, self.kernel_size, self.stride)
        return self.conv(x)


class BnReluDrop(nn.Module):
    """Port of `_bn_relu`: BatchNorm -> Activation -> (Dropout if >0)."""

    def __init__(self, num_features, dropout=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResNetBlock(nn.Module):
    """Port of `resnet_block`."""

    def __init__(
        self,
        in_ch,
        num_filters,
        subsample_length,
        block_index,
        conv_filter_length,
        conv_num_skip,
        conv_dropout,
        conv_increase_channels_at,
    ):
        super().__init__()
        self.subsample_length = subsample_length
        self.block_index = block_index
        self.conv_increase_channels_at = conv_increase_channels_at
        self.zero_pad = (block_index % conv_increase_channels_at == 0) and block_index > 0
        self.shortcut_pool = (
            nn.MaxPool1d(kernel_size=subsample_length, stride=subsample_length)
            if subsample_length > 1
            else nn.Identity()
        )

        layers = []
        ch = in_ch
        for i in range(conv_num_skip):
            if not (block_index == 0 and i == 0):
                layers.append(BnReluDrop(ch, dropout=conv_dropout if i > 0 else 0.0))
            stride = subsample_length if i == 0 else 1
            layers.append(ConvSame1d(ch, num_filters, conv_filter_length, stride))
            ch = num_filters
        self.main_path = nn.ModuleList(layers)

    def forward(self, x):
        shortcut = self.shortcut_pool(x)
        if self.zero_pad:
            zeros = torch.zeros_like(shortcut)
            shortcut = torch.cat([shortcut, zeros], dim=1)  # channel-doubling zero pad

        h = x
        for layer in self.main_path:
            h = layer(h)
        return shortcut + h


class DeepHeartECGNet(nn.Module):
    """Faithful port of `build_network()` (resnet path, `is_regular_conv=
    False`, the repo's default and the config used for `examples/cinc17`)."""

    def __init__(
        self,
        num_categories=4,
        conv_subsample_lengths=(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2),
        conv_filter_length=16,
        conv_num_filters_start=32,
        conv_dropout=0.2,
        conv_num_skip=2,
        conv_increase_channels_at=4,
    ):
        super().__init__()
        self.initial_conv = ConvSame1d(1, conv_num_filters_start, conv_filter_length, 1)
        self.initial_bn_relu = BnReluDrop(conv_num_filters_start, dropout=0.0)

        blocks = []
        ch = conv_num_filters_start
        for index, subsample_length in enumerate(conv_subsample_lengths):
            num_filters = 2 ** int(index / conv_increase_channels_at) * conv_num_filters_start
            blocks.append(
                ResNetBlock(
                    ch,
                    num_filters,
                    subsample_length,
                    index,
                    conv_filter_length,
                    conv_num_skip,
                    conv_dropout,
                    conv_increase_channels_at,
                )
            )
            ch = num_filters
        self.blocks = nn.ModuleList(blocks)
        self.final_bn_relu = BnReluDrop(ch, dropout=0.0)

        self.output_dense = nn.Linear(ch, num_categories)

    def forward(self, x):  # x: [N, 1, T] raw ECG signal (single lead)
        h = self.initial_conv(x)
        h = self.initial_bn_relu(h)
        for block in self.blocks:
            h = block(h)
        h = self.final_bn_relu(h)
        h = h.transpose(1, 2)  # [N, T', C] for per-timestep Dense (TimeDistributed)
        h = self.output_dense(h)
        return F.softmax(h, dim=-1)


def build_deepheart_ecg():
    return DeepHeartECGNet()


def example_input_deepheart_ecg():
    return torch.randn(2, 1, 2560)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepHeart-ECG", "build_deepheart_ecg", "example_input_deepheart_ecg", 2017, "ported"),
]
