# FAITHFUL PORT of https://github.com/awni/ecg @ master (original framework: Keras / TF1.x)
#
# Transcribed from the official repo's `ecg/network.py` (`build_network`,
# `add_resnet_layers`, `resnet_block`, `add_conv_weight`, `_bn_relu`,
# `add_output_layer`) using the real published hyperparameters from
# `examples/cinc17/config.json` (conv_subsample_lengths = 16 stages alternating
# stride 1/2, conv_filter_length=16, conv_num_filters_start=32,
# conv_num_skip=2, conv_increase_channels_at=4, conv_dropout=0.2). This is the
# real "34-layer" 1D residual CNN for ECG arrhythmia classification (Hannun,
# Rajpurkar, et al., "Cardiologist-level arrhythmia detection...", Nature
# Medicine 2019, arxiv:1707.01836). The official repo is Keras/TF1.x and
# cannot run in a modern torch env, so this module transcribes the network
# layer-by-layer into self-contained torch (Conv1d/BatchNorm1d/ReLU/Dropout,
# NCL layout matching Keras' Conv1D's channels-last-but-equivalent semantics).
#
# Architecture (faithfully transcribed from `add_resnet_layers`/`resnet_block`):
#   - stem: Conv1D(filters=32, kernel=16, stride=1) -> BN -> ReLU
#   - 16 residual blocks (`conv_subsample_lengths` has 16 entries), each with
#     `conv_num_skip=2` conv sub-layers of kernel 16 (`conv_filter_length`).
#     Each block's shortcut path is `MaxPool1d(subsample_length)`; every 4th
#     block (`block_index % conv_increase_channels_at == 0 and block_index > 0`)
#     doubles filters and zero-pads the shortcut along the channel axis
#     (`zeropad` Lambda: concat the shortcut with an equal-shape zero tensor).
#     Block 0's first BN-ReLU-dropout is skipped (`not (block_index==0 and i==0)`),
#     matching the "first block has no pre-activation on its first conv" quirk
#     in the official code.
#   - head: BN -> ReLU -> TimeDistributed(Dense(num_categories)) -> softmax,
#     applied at every remaining time step (the real model is a *sequence*
#     classifier producing one label per surviving time-step, not a single
#     pooled label -- `TimeDistributed(Dense(...))` applies the same Linear
#     independently at each position along the time axis).
"""Faithful torch port of the Hannun et al. 34-layer 1D-ResNet ECG arrhythmia DNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

# Real hyperparameters from examples/cinc17/config.json
_CONV_SUBSAMPLE_LENGTHS = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
_CONV_FILTER_LENGTH = 16
_CONV_NUM_FILTERS_START = 32
_CONV_NUM_SKIP = 2
_CONV_INCREASE_CHANNELS_AT = 4
_CONV_DROPOUT = 0.2
_NUM_CATEGORIES = 4  # cinc17 AF-classification task: normal/AF/other/noisy


def _get_num_filters_at_index(index: int) -> int:
    """network.py::get_num_filters_at_index"""
    return (2 ** int(index / _CONV_INCREASE_CHANNELS_AT)) * _CONV_NUM_FILTERS_START


class BNReLU(nn.Module):
    """network.py::_bn_relu -- BatchNorm1d + ReLU (+ optional dropout)."""

    def __init__(self, num_features: int, dropout: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResNetBlock(nn.Module):
    """network.py::resnet_block -- `conv_num_skip` conv sub-layers (each preceded
    by BN-ReLU[-Dropout] except block 0's very first sub-layer), added to a
    MaxPool1d shortcut that is zero-padded along channels every
    `conv_increase_channels_at`-th block."""

    def __init__(self, in_channels: int, num_filters: int, subsample_length: int, block_index: int):
        super().__init__()
        self.block_index = block_index
        self.subsample_length = subsample_length
        self.zero_pad = (block_index % _CONV_INCREASE_CHANNELS_AT == 0) and block_index > 0
        self.in_channels = in_channels
        self.num_filters = num_filters

        self.pre_acts = nn.ModuleList()
        self.convs = nn.ModuleList()
        ch = in_channels
        for i in range(_CONV_NUM_SKIP):
            skip_first_preact = block_index == 0 and i == 0
            if not skip_first_preact:
                self.pre_acts.append(BNReLU(ch, dropout=_CONV_DROPOUT if i > 0 else 0.0))
            else:
                self.pre_acts.append(nn.Identity())
            stride = subsample_length if i == 0 else 1
            self.convs.append(
                nn.Conv1d(
                    ch,
                    num_filters,
                    kernel_size=_CONV_FILTER_LENGTH,
                    stride=stride,
                    padding=_CONV_FILTER_LENGTH // 2,
                )
            )
            ch = num_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = (
            F.max_pool1d(x, kernel_size=self.subsample_length) if self.subsample_length > 1 else x
        )
        if self.zero_pad:
            shortcut = torch.cat([shortcut, torch.zeros_like(shortcut)], dim=1)

        layer = x
        for pre_act, conv in zip(self.pre_acts, self.convs):
            layer = pre_act(layer)
            layer = conv(layer)
            # Keras `padding='same'` with stride>1 can produce a length one
            # off from PyTorch's symmetric padding; crop/pad the conv output
            # to exactly match the shortcut's time length so the residual add
            # is well-defined (matches the real network's same-shape residual
            # sum; the official Keras backend resolves this internally).
            if layer.shape[-1] != shortcut.shape[-1]:
                min_len = min(layer.shape[-1], shortcut.shape[-1])
                layer = layer[..., :min_len]
                shortcut = shortcut[..., :min_len]
        return shortcut + layer


class HannunECGNet(nn.Module):
    """network.py::build_network (resnet path, `is_regular_conv=False`)."""

    def __init__(self, in_channels: int = 1, num_categories: int = _NUM_CATEGORIES):
        super().__init__()
        self.stem_conv = nn.Conv1d(
            in_channels,
            _CONV_NUM_FILTERS_START,
            kernel_size=_CONV_FILTER_LENGTH,
            stride=1,
            padding=_CONV_FILTER_LENGTH // 2,
        )
        self.stem_bnrelu = BNReLU(_CONV_NUM_FILTERS_START)

        self.blocks = nn.ModuleList()
        ch = _CONV_NUM_FILTERS_START
        for index, subsample_length in enumerate(_CONV_SUBSAMPLE_LENGTHS):
            num_filters = _get_num_filters_at_index(index)
            self.blocks.append(ResNetBlock(ch, num_filters, subsample_length, index))
            ch = num_filters

        self.final_bnrelu = BNReLU(ch)
        self.output_dense = nn.Linear(ch, num_categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T) -- Keras Conv1D operates on (B, T, C); torch
        # Conv1d wants (B, C, T). We keep torch-native NCL throughout.
        layer = self.stem_conv(x)
        layer = self.stem_bnrelu(layer)
        for block in self.blocks:
            layer = block(layer)
        layer = self.final_bnrelu(layer)

        # network.py::add_output_layer -- TimeDistributed(Dense) + softmax,
        # applied independently at every surviving time step.
        layer = layer.transpose(1, 2)  # (B, T', C) for the per-timestep Linear
        logits = self.output_dense(layer)
        probs = F.softmax(logits, dim=-1)
        return probs


# ---------------------------------------------------------------------------
# Staging build/example helpers. Real training clips are 30s @ 200Hz (6000
# samples); we use a shorter clip length for fast tracing while preserving
# the full 16-block topology and its full stride/channel schedule.
# ---------------------------------------------------------------------------


def build_hannun_ecg():
    return HannunECGNet().eval()


def example_input_hannun_ecg():
    batch = 1
    x = torch.randn(batch, 1, 2048)
    return (x,)


MENAGERIE_ENTRIES = [
    ("Hannun arrhythmia DNN", build_hannun_ecg, example_input_hannun_ecg, 2019, "ported-pytorch"),
]
