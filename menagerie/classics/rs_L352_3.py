# FAITHFUL PORT of Seb-Good/deepecg @ master (original framework: TensorFlow 1.x, graph mode)
#
# Source: deepecg/training/networks/deep_ecg_v7.py::DeepECGV7 (+ layers.py helpers), the
# flagship "Inception-V4 and ResNet inspired" WaveNet-style network. Confirmed as the
# actually-trained network (not the README's simpler 13-layer DeepECGV1) via
# deepecg/training/notebooks/training/disc/1_train_model.ipynb, which sets
# `network_name = 'DeepECGV7'` (save path 'wavenet_2').
#
# Architecture transcribed layer-for-layer from `DeepECGV7.inference`:
#   layer_1: Conv1D(kernel=3, filters=128, dilation=1, no activation, no bias)
#   layer_2..layer_10: WaveNet-style gated residual blocks at dilation rates
#     2, 4, 8, 16, 32, 64, 128, 256, 512 -- each block computes
#     tanh(conv_filt(x)) * sigmoid(conv_gate(x)), then a 1x1 conv "res" branch
#     (added back to the block input as an identity skip-connection) and a 1x1
#     conv "skip" branch collected into a running list. layer_10 (dilation=512)
#     has res=False (per `_residual_block(..., res=False, skip=True)` in the
#     original), i.e. contributes only a skip output, matching this port.
#   All 9 block skip outputs are summed (`tf.add_n`), then ReLU -> Dropout(0.3) ->
#     Conv1D(k=3, filters=256, relu, no bias) -> Dropout(0.3) ->
#     Conv1D(k=3, filters=512, relu, no bias) -> Dropout(0.3) ->
#     GlobalAveragePooling (mean over time) -> Dense(classes, no bias) == logits.
# Class-activation-map computation (`_get_cams`/`_compute_cam`) is a training-time
# diagnostic that reuses the final conv's weight matrix and does not affect the
# forward classification graph; it is not part of the traced module.
# TF's `tf.layers.conv1d(..., padding='SAME')` matches torch `padding=dilation*(k-1)//2`
# for the odd kernel_size=3 used throughout.
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class _ResidualBlock(nn.Module):
    """WaveNet-style gated residual block, port of
    deep_ecg_v7.py::DeepECGV7._residual_block."""

    def __init__(
        self, in_ch, conv_filts, res_filts, skip_filts, kernel_size, dilation, res=True, skip=True
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv_filt = nn.Conv1d(
            in_ch, conv_filts, kernel_size, padding=pad, dilation=dilation, bias=False
        )
        self.conv_gate = nn.Conv1d(
            in_ch, conv_filts, kernel_size, padding=pad, dilation=dilation, bias=False
        )
        self.has_res = res
        self.has_skip = skip
        if res:
            self.conv_res = nn.Conv1d(conv_filts, res_filts, 1, bias=False)
        if skip:
            self.conv_skip = nn.Conv1d(conv_filts, skip_filts, 1, bias=False)

    def forward(self, x):
        conv_filt = torch.tanh(self.conv_filt(x))
        conv_gate = torch.sigmoid(self.conv_gate(x))
        activation = conv_filt * conv_gate
        res_out = None
        skip_out = None
        if self.has_res:
            res_out = self.conv_res(activation) + x
        if self.has_skip:
            skip_out = self.conv_skip(activation)
        return res_out, skip_out


class DeepECGV7(nn.Module):
    """Port of deepecg/training/networks/deep_ecg_v7.py::DeepECGV7.inference."""

    def __init__(self, channels=1, classes=4):
        super().__init__()
        kernel_size = 3
        conv_filts = 128
        res_filts = 128
        skip_filts = 128
        dilations = [2, 4, 8, 16, 32, 64, 128, 256, 512]  # layer_2 .. layer_10

        self.layer1_conv = nn.Conv1d(
            channels, res_filts, kernel_size, padding=1, dilation=1, bias=False
        )

        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(
                    res_filts,
                    conv_filts,
                    res_filts,
                    skip_filts,
                    kernel_size,
                    d,
                    res=(i != len(dilations) - 1),  # layer_10 (last) is res=False, skip=True
                    skip=True,
                )
                for i, d in enumerate(dilations)
            ]
        )

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(skip_filts, 256, kernel_size, padding=1, bias=False)
        self.dropout2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(256, 512, kernel_size, padding=1, bias=False)
        self.dropout3 = nn.Dropout(0.3)
        self.logits = nn.Linear(512, classes, bias=False)

    def forward(self, x):
        # x: (B, channels, length) -- torch Conv1d channel-first convention;
        # TF code uses (B, length, channels) but the op sequence is identical.
        net = self.layer1_conv(x)
        skips = []
        for block in self.blocks:
            res_out, skip_out = block(net)
            skips.append(skip_out)
            if res_out is not None:
                net = res_out

        output = torch.stack(skips, dim=0).sum(dim=0)
        output = self.relu(output)
        output = self.dropout1(output)
        output = self.relu(self.conv1(output))
        output = self.dropout2(output)
        output = self.relu(self.conv2(output))
        output = self.dropout3(output)

        gap = output.mean(dim=2)
        logits = self.logits(gap)
        return logits


def build_deepecg_v7():
    return DeepECGV7(channels=1, classes=4)


def example_input_deepecg_v7():
    # Real training used length=12000 (30s @ 400Hz); a small representative length
    # is used here so the dilated-conv stack (max dilation 512) traces quickly.
    return torch.randn(1, 1, 4096)


MENAGERIE_ENTRIES = [
    ("DeepECG-V7-WaveNet", build_deepecg_v7, example_input_deepecg_v7, 2018, "ported-pytorch"),
]
