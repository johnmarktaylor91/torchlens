# SOURCE: vendored from Kylin9511/CRNet @ master
# (models/crnet.py, copied verbatim; the `from utils import logger` import call in
# CRNet.__init__ is dropped in favor of a no-op since the original logger module
# has repo-relative config side effects unrelated to the architecture)
#
# CRNet: a CSI (channel state information) feedback compression network for massive
# MIMO. Dual-path convolutional encoder (asymmetric 1x9/9x1 branch + plain 3x3
# branch) compresses a (2, 32, 32) CSI image to a low-dim code via an FC bottleneck,
# then a decoder FC + convolutional CRBlocks (two parallel asymmetric-kernel paths
# fused by a 1x1 conv, residual) reconstructs the CSI image. Only base-lib deps: torch.

from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models/crnet.py (verbatim) ----


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
                            in_planes,
                            out_planes,
                            kernel_size,
                            stride,
                            padding=padding,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_planes)),
                ]
            )
        )


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv3x3", ConvBN(2, 7, 3)),
                    ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv1x9", ConvBN(7, 7, [1, 9])),
                    ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv9x1", ConvBN(7, 7, [9, 1])),
                ]
            )
        )
        self.path2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1x5", ConvBN(2, 7, [1, 5])),
                    ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv5x1", ConvBN(7, 7, [5, 1])),
                ]
            )
        )
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class CRNet(nn.Module):
    def __init__(self, reduction=4):
        super(CRNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32  # noqa: F841 (verbatim from upstream)
        self.encoder1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
                    ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
                    ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
                ]
            )
        )
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(
            OrderedDict(
                [
                    ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv1x1_bn", ConvBN(4, 2, 1)),
                    ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ]
            )
        )
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        decoder = OrderedDict(
            [
                ("conv5x5_bn", ConvBN(2, 2, 5)),
                ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ("CRBlock1", CRBlock()),
                ("CRBlock2", CRBlock()),
            ]
        )
        self.decoder_feature = nn.Sequential(decoder)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()

        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))

        out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out


def crnet(reduction=4):
    r"""Create a proposed CRNet.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of CRNet
    """

    model = CRNet(reduction=reduction)
    return model


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_crnet():
    """CRNet at the repo's own CSI-image size (32x32x2, architecture unmodified)."""
    torch.manual_seed(0)
    model = crnet(reduction=4)
    model.eval()
    return model


def example_input_crnet():
    torch.manual_seed(0)
    return torch.rand(2, 2, 32, 32)


MENAGERIE_ENTRIES = [
    ("CRNet", "build_crnet", "example_input_crnet", 2020, MENAGERIE_ZOO),
]
