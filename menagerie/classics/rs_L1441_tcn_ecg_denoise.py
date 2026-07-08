# SOURCE: vendored from locuslab/TCN @ master
# https://raw.githubusercontent.com/locuslab/TCN/master/TCN/tcn.py
#
# "TCN-based ECG-Denoising" (queue candidate: TCN architecture applicable to
# ECG denoising/regression, e.g. the ECG-TCN paper arxiv:2103.13740, which uses
# the same causal dilated Temporal Convolutional Network backbone popularized by
# locuslab/TCN, "An Empirical Evaluation of Generic Convolutional and Recurrent
# Networks for Sequence Modeling"). The reusable, task-agnostic architecture is
# `TemporalConvNet`/`TemporalBlock`/`Chomp1d` from locuslab/TCN's `TCN/tcn.py` --
# a stack of causal dilated Conv1d residual blocks with weight-normalized
# convolutions, exponentially increasing dilation per stack level, and a
# 1x1-conv downsample skip when channel counts change. Transcribed verbatim,
# no architectural changes. Applied here directly to a 1-channel signal input
# (e.g. a single-lead ECG waveform) matching the "sequence in, sequence out"
# usage pattern of TCN-based ECG denoising works built on this backbone.

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNDenoiser(nn.Module):
    """1-in/1-out sequence-to-sequence wrapper around ``TemporalConvNet``,
    matching the ECG-denoising usage pattern: a single-lead noisy ECG signal
    goes in, a denoised single-lead signal comes out (regression, not
    classification -- no final linear/softmax head is added by the real
    TCN backbone itself)."""

    def __init__(self, num_channels, kernel_size=7, dropout=0.0):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.out = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        y = self.tcn(x)
        return self.out(y)


def build_tcn_ecg_denoise():
    torch.manual_seed(0)
    model = TCNDenoiser(num_channels=[16, 16, 16, 16], kernel_size=7, dropout=0.0)
    model.eval()
    return model


def example_input_tcn_ecg_denoise():
    torch.manual_seed(0)
    # (batch, channels=1 lead, time)
    return torch.randn(2, 1, 512)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "TCN-ECG-Denoising",
        "build_tcn_ecg_denoise",
        "example_input_tcn_ecg_denoise",
        2018,
        MENAGERIE_ZOO,
    ),
]
