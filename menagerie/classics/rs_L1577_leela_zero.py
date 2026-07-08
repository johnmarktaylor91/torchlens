# FAITHFUL PORT of https://github.com/leela-zero/leela-zero @ next (original framework: TensorFlow 1.x)
# (training/tf/tfprocess.py: TFProcess.conv_block / TFProcess.residual_block /
#  TFProcess.construct_net, lines 558-642)
#
# Leela Zero (an open-source, from-scratch reimplementation of "Mastering the
# Game of Go without Human Knowledge" / AlphaGo Zero, Silver et al. 2017). The
# engine itself is C++ (src/Network.cpp) with no Python model class; the
# reference architecture that the C++ engine's weights implement is defined by
# the project's own TensorFlow 1.x training code (`training/tf/tfprocess.py`),
# built with raw `tf.nn`/manual `weight_variable` graph construction (no Keras
# class, so it cannot be vendored as-is or run in a base torch env). This is a
# faithful line-by-line transcription of that TF1 graph into a self-contained
# torch nn.Module: an input conv-block, a tower of `residual_blocks` standard
# (non-SE) residual blocks -- conv3x3-bn-relu-conv3x3-bn-add(skip)-relu -- and
# two heads reading off the tower: a policy head (1x1 conv to 2 channels ->
# flatten -> FC to board_size**2 + 1 for the pass move) and a value head (1x1
# conv to 1 channel -> flatten -> FC(256) -> relu -> FC(1) -> tanh). Board size
# is a free faithful parameter (the real net is 19x19 Go with 18 input planes);
# kept configurable and reduced here purely for a fast trace.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class _ConvBlock(nn.Module):
    """TFProcess.conv_block: conv -> batchnorm -> relu."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _ResidualBlock(nn.Module):
    """TFProcess.residual_block: two 3x3 conv-bn stages with a skip
    connection, matching the TF1 graph exactly (relu after the first
    conv-bn, add-then-relu after the second)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        orig = x
        net = F.relu(self.bn1(self.conv1(x)))
        net = self.bn2(self.conv2(net))
        net = net + orig
        return F.relu(net)


class LeelaZeroNet(nn.Module):
    """TFProcess.construct_net: input conv-block, residual tower, policy
    head, value head."""

    def __init__(self, input_channels=18, board_size=19, residual_blocks=2, residual_filters=32):
        super().__init__()
        self.input_channels = input_channels
        self.board_size = board_size
        n = board_size * board_size

        self.input_conv = _ConvBlock(input_channels, residual_filters, kernel_size=3)
        self.tower = nn.ModuleList(
            [_ResidualBlock(residual_filters) for _ in range(residual_blocks)]
        )

        # Policy head
        self.policy_conv = _ConvBlock(residual_filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * n, n + 1)

        # Value head
        self.value_conv = _ConvBlock(residual_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(n, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, planes):
        batch = planes.size(0)
        x_planes = planes.view(batch, self.input_channels, self.board_size, self.board_size)

        flow = self.input_conv(x_planes)
        for block in self.tower:
            flow = block(flow)

        # Policy head
        conv_pol = self.policy_conv(flow)
        h_conv_pol_flat = conv_pol.reshape(batch, -1)
        policy_logits = self.policy_fc(h_conv_pol_flat)

        # Value head
        conv_val = self.value_conv(flow)
        h_conv_val_flat = conv_val.reshape(batch, -1)
        h_fc2 = F.relu(self.value_fc1(h_conv_val_flat))
        value = torch.tanh(self.value_fc2(h_fc2))

        return policy_logits, value


def build_leela_zero():
    return LeelaZeroNet(input_channels=18, board_size=9, residual_blocks=2, residual_filters=16)


def example_input_leela_zero():
    return (torch.randn(2, 18, 9, 9),)


MENAGERIE_ENTRIES = [
    ("Leela Zero", "build_leela_zero", "example_input_leela_zero", 2017, "ported-pytorch"),
]
