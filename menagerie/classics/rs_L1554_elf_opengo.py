# SOURCE: vendored from pytorch/ELF @ master
# https://github.com/pytorch/ELF
# File: src_py/elfgames/go/df_model3.py (Block, GoResNet, Model_PolicyValue)
#
# ELF OpenGo (Tian et al. 2019) is Facebook's open-source AlphaGo-Zero-style
# Go engine built on the ELF (Extensive, Lightweight, Flexible) game-research
# platform. Its network is `Model_PolicyValue` in df_model3.py: an initial
# conv layer, a stack of residual `Block`s (conv-bn-relu, conv-bn, + skip,
# relu -- following the AlphaGo Zero architecture), then a policy head
# (1x1 conv -> linear -> log-softmax over board positions + pass) and a value
# head (1x1 conv -> linear -> relu -> linear -> tanh), exactly as described in
# the class docstring ("Network structure of AlphaGo Zero"). This module
# vendors the real `Block`, `GoResNet`, and the policy/value network body of
# `Model_PolicyValue` verbatim (conv/bn/relu layer structure, channel counts,
# forward-pass order unchanged). Only the `elf.options.PyOptionSpec` /
# `rlpytorch.Model` config-injection scaffolding and the CUDA /
# DataParallel / torch.distributed bring-up code (irrelevant to the network's
# architecture, and none of it is installable outside the ELF package) were
# replaced with plain constructor arguments and removed respectively.

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Block(nn.Module):
    """Residual block: conv-bn-relu, conv-bn, + skip, relu. (df_model3.py Block)"""

    def __init__(self, dim=128, bn=True, bn_momentum=0.1, bn_eps=1e-5, leaky_relu=False):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1) if leaky_relu else nn.ReLU()
        self.dim = dim
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.conv_lower = self._conv_layer()
        self.conv_upper = self._conv_layer(relu=False)

    def _conv_layer(self, input_channel=None, output_channel=None, kernel=3, relu=True):
        if input_channel is None:
            input_channel = self.dim
        if output_channel is None:
            output_channel = self.dim

        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel, padding=(kernel // 2)))
        if self.bn:
            layers.append(
                nn.BatchNorm2d(output_channel, momentum=(self.bn_momentum or None), eps=self.bn_eps)
            )
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, s):
        s1 = self.conv_lower(s)
        s1 = self.conv_upper(s1)
        s1 = s1 + s
        s = self.relu(s1)
        return s


class GoResNet(nn.Module):
    """Stack of residual Blocks. (df_model3.py GoResNet)"""

    def __init__(
        self, num_block=20, dim=128, bn=True, bn_momentum=0.1, bn_eps=1e-5, leaky_relu=False
    ):
        super().__init__()
        self.blocks = []
        for _ in range(num_block):
            self.blocks.append(
                Block(dim=dim, bn=bn, bn_momentum=bn_momentum, bn_eps=bn_eps, leaky_relu=leaky_relu)
            )
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, s):
        return self.resnet(s)


class Model_PolicyValue(nn.Module):
    """AlphaGo-Zero-style policy/value network. (df_model3.py Model_PolicyValue)

    Network structure of AlphaGo Zero:
    https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html
    """

    def __init__(
        self,
        board_size,
        num_planes,
        num_block=20,
        dim=128,
        bn=True,
        bn_momentum=0.1,
        bn_eps=1e-5,
        leaky_relu=False,
    ):
        super().__init__()

        self.board_size = board_size
        self.num_planes = num_planes
        self.dim = dim
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps

        # Simple method. multiple conv layers.
        self.relu = nn.LeakyReLU(0.1) if leaky_relu else nn.ReLU()
        last_planes = self.num_planes

        self.init_conv = self._conv_layer(last_planes)

        self.pi_final_conv = self._conv_layer(self.dim, 2, 1)
        self.value_final_conv = self._conv_layer(self.dim, 1, 1)

        d = self.board_size**2

        # Plus 1 for pass.
        self.pi_linear = nn.Linear(d * 2, d + 1)
        self.value_linear1 = nn.Linear(d, 256)
        self.value_linear2 = nn.Linear(256, 1)

        # Softmax as the final layer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.resnet = GoResNet(
            num_block=num_block,
            dim=dim,
            bn=bn,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            leaky_relu=leaky_relu,
        )

    def _conv_layer(self, input_channel=None, output_channel=None, kernel=3, relu=True):
        if input_channel is None:
            input_channel = self.dim
        if output_channel is None:
            output_channel = self.dim

        layers = []
        layers.append(nn.Conv2d(input_channel, output_channel, kernel, padding=(kernel // 2)))
        if self.bn:
            layers.append(
                nn.BatchNorm2d(output_channel, momentum=(self.bn_momentum or None), eps=self.bn_eps)
            )
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, s):
        s = self.init_conv(s)
        s = self.resnet(s)

        d = self.board_size**2

        pi = self.pi_final_conv(s)
        pi = self.pi_linear(pi.view(-1, d * 2))
        logpi = self.logsoftmax(pi)
        pi = logpi.exp()

        v = self.value_final_conv(s)
        v = self.relu(self.value_linear1(v.view(-1, d)))
        v = self.value_linear2(v)
        v = self.tanh(v)

        return pi, v


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


def build_elf_opengo():
    torch.manual_seed(0)
    model = Model_PolicyValue(board_size=9, num_planes=18, num_block=2, dim=16)
    model.eval()
    return model


def example_input_elf_opengo():
    torch.manual_seed(0)
    return torch.randn(1, 18, 9, 9)


MENAGERIE_ENTRIES = [
    (
        "ELF OpenGo (AlphaGo-Zero-style policy/value ResNet)",
        "build_elf_opengo",
        "example_input_elf_opengo",
        2019,
        MENAGERIE_ZOO,
    ),
]
