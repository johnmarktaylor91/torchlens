# SOURCE: vendored from https://github.com/Kylin9511/ACRNet @ main
#   (model/acrnet.py), MIT License, Copyright (c) 2021 Kylin Lu.
# Paper: Lu, Chen et al., "Binarized Aggregated Network with Quantization:
# Flexible Deep Learning Deployment for CSI Feedback in Massive MIMO System"
# (arXiv:2105.00354). This is the repo's real, complete FP32 inference
# architecture for ACRNet -- a channel-state-information (CSI) feedback
# autoencoder for massive-MIMO systems (encodes a 2x32x32 complex channel
# matrix down to a compressed code and reconstructs it), reproducing the
# paper's stated results on the COST2100 channel model.
#
# The ONLY change from the real `model/acrnet.py` is dropping the repo-internal
# `from utils import logger` import and its single `logger.info(...)` call in
# `ACRNet.__init__` (informational-only side effect from the original repo's
# own logging module, not part of the architecture) -- everything else
# (class/method names, layer definitions, forward logic, weight init) is
# unmodified.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from collections import OrderedDict

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

__all__ = ["acrnet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels=in_planes,
                            out_channels=out_planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_planes)),
                ]
            )
        )


class ACRDecoderBlock(nn.Module):
    r"""Inverted residual with extensible width and group conv"""

    def __init__(self, expansion):
        super(ACRDecoderBlock, self).__init__()
        width = 8 * expansion
        self.conv1_bn = ConvBN(2, width, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv2_bn = ConvBN(width, width, 7, groups=4 * expansion)
        self.prelu2 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv3_bn = ConvBN(width, 2, [9, 1])
        self.prelu3 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.prelu1(self.conv1_bn(x))
        residual = self.prelu2(self.conv2_bn(residual))
        residual = self.conv3_bn(residual)

        return self.prelu3(identity + residual)


class ACREncoderBlock(nn.Module):
    def __init__(self):
        super(ACREncoderBlock, self).__init__()
        self.conv_bn1 = ConvBN(2, 2, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv_bn2 = ConvBN(2, 2, [9, 1])
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.prelu1(self.conv_bn1(x))
        residual = self.conv_bn2(residual)

        return self.prelu2(identity + residual)


class ACRNet(nn.Module):
    def __init__(self, in_channels=2, reduction=4, expansion=1):
        super(ACRNet, self).__init__()
        total_size = 2048

        self.encoder_feature = nn.Sequential(
            OrderedDict(
                [
                    ("conv5x5_bn", ConvBN(in_channels, 2, 5)),
                    ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
                    ("ACREncoderBlock1", ACREncoderBlock()),
                    ("ACREncoderBlock2", ACREncoderBlock()),
                ]
            )
        )
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder_feature = nn.Sequential(
            OrderedDict(
                [
                    ("conv5x5_bn", ConvBN(2, in_channels, 5)),
                    ("prelu", nn.PReLU(num_parameters=2, init=0.3)),
                    ("ACRDecoderBlock1", ACRDecoderBlock(expansion=expansion)),
                    ("ACRDecoderBlock2", ACRDecoderBlock(expansion=expansion)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()

        out = self.encoder_feature(x)
        out = self.encoder_fc(out.view(n, -1))

        out = self.decoder_fc(out)
        out = self.decoder_feature(out.view(n, c, h, w))

        return out


def acrnet(reduction=4, expansion=1):
    r"""Create an ACRNet architecture."""
    model = ACRNet(reduction=reduction, expansion=expansion)
    return model


def build_acrnet():
    return acrnet(reduction=4, expansion=1)


def example_input_acrnet():
    return torch.rand(4, 2, 32, 32)  # (N, real/imag, subcarriers, antennas)


MENAGERIE_ENTRIES = [
    ("ACRNet", "build_acrnet", "example_input_acrnet", 2021, MENAGERIE_ZOO),
]
