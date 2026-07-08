# SOURCE: vendored from SIJIEJI/CLNet @ main
# https://raw.githubusercontent.com/SIJIEJI/CLNet/main/models/clnet.py
#
# CLNet (Ji & Li, "CLNet: Complex Input Lightweight Neural Network Designed for
# Massive MIMO CSI Feedback", IEEE WCL 2021): a CSI-feedback autoencoder for
# massive-MIMO channel state information compression/reconstruction, built from
# ConvBN blocks, a dual-path (3x3-factorized + 1x5/5x1-factorized) CRBlock refine
# unit, and channel/spatial attention (SELayer / SpatialGate, CBAM-style). All
# classes (`ConvBN`, `CRBlock`, `hsigmoid`, `BasicConv`, `ChannelPool`, `SpatialGate`,
# `SELayer`, `Encoder`, `Decoder`, `CLNet`, `clnet`) are copied verbatim from the
# official repo's models/clnet.py. The only change is dropping the
# `from utils import logger` import and the single `logger.info(...)` call in
# `CLNet.__init__` (a training-run log line with no effect on the architecture).
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

__all__ = ["clnet"]


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
                    ("conv1x3", ConvBN(7, 7, [1, 3])),
                    ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv3x1", ConvBN(7, 7, [3, 1])),
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


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Encoder(nn.Module):
    def __init__(self, reduction=4):
        super(Encoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32  # noqa: F841 (verbatim from source)
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
        self.encoder2 = ConvBN(in_channel, 32, 1)
        self.encoder_conv = nn.Sequential(
            OrderedDict(
                [
                    ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                    ("conv1x1_bn", ConvBN(34, 2, 1)),
                    ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
                ]
            )
        )
        self.sa = SpatialGate()
        self.se = SELayer(32)
        self.replace_efc = nn.Conv1d(total_size, total_size // reduction, 1)

    def forward(self, x):
        n, c, h, w = x.detach().size()
        encode1 = self.encoder1(x)
        encode1 = self.sa(encode1)
        encode2 = self.encoder2(x)
        encode2 = self.se(encode2)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)

        out = out.view(n, -1)
        out = out.unsqueeze(2)  # [1,2048,1]
        out = self.replace_efc(out)  # [1,2048/cr,1]

        return out


class Decoder(nn.Module):
    def __init__(self, reduction=4):
        super(Decoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32  # noqa: F841 (verbatim from source)
        self.replace_dfc = nn.ConvTranspose1d(total_size // reduction, total_size, 1)
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
        self.hsig = hsigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c, h, w = 2, 32, 32

        out = self.replace_dfc(x)  # [1,2048,1]
        out = out.view(-1, c, h, w)

        out = self.decoder_feature(out)
        out = self.hsig(out)

        return out


class CLNet(nn.Module):
    def __init__(self, reduction=4):
        super(CLNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32  # noqa: F841 (verbatim from source)
        self.encoder = Encoder(reduction)
        self.decoder = Decoder(reduction)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def clnet(reduction=4):
    r"""Create a proposed CLNet.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of CLNet
    """

    model = CLNet(reduction=reduction)
    return model


def build_clnet():
    # Matches the real constructor call used across the repo's configs (e.g.
    # main.py's default `--reduction 4`); the encoder/decoder internally hard-code
    # the COST2100 CSI matrix size (2, 32, 32) -> total_size=2048, so the input
    # shape below is fixed by the architecture, not a menagerie-scale shrink.
    return clnet(reduction=4)


def example_input_clnet():
    # Real-imaginary CSI matrix: (batch, 2, 32, 32) per the Encoder/Decoder's
    # hard-coded (in_channel, w, h) = (2, 32, 32).
    torch.manual_seed(0)
    return (torch.rand(2, 2, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "CLNet (Massive MIMO CSI Feedback)",
        "build_clnet",
        "example_input_clnet",
        2021,
        "vendored-pytorch",
    ),
]
