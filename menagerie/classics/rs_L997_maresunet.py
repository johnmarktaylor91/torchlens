# SOURCE: vendored from lironui/MAResU-Net @ 3a3564ff753a8f0769737533bbf4ff3dad1a1f7b
# (MAResUNet.py)
#
# MAResU-Net (Multi-stage Attention ResU-Net): ResNet34 encoder (torchvision backbone)
# feeding a U-Net-style decoder where every decoder stage is preceded by a PAM_CAM_Layer
# -- a linear-attention Position Attention Module (PAM, kernel-trick softmax-free spatial
# attention) fused with a Channel Attention Module (CAM) via parallel conv branches summed
# together. conv3otherRelu/PAM_Module/CAM_Module/PAM_CAM_Layer/DecoderBlock/MAResUNet
# below are the REAL model code from MAResUNet.py, copied verbatim (only base-lib deps:
# torch, torchvision). `pretrained=False` is used in build_maresunet() purely to avoid a
# network weights download for tracing -- the architecture is untouched.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Parameter, Softmax
from torchvision import models

MENAGERIE_ZOO = "vendored-pytorch"

nonlinearity = partial(F.relu, inplace=True)


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), "kernel_size is not in (int, tuple)!"

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), "stride is not in (int, tuple)!"

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), "padding is not in (int, tuple)!"

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        ),
        nn.ReLU(inplace=True),  # inplace=True
    )


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(
            in_channels=in_places, out_channels=in_places // scale, kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_places, out_channels=in_places // scale, kernel_size=1
        )
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (
            width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        )
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum("bmn, bcn->bmc", K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.conv1 = conv3otherRelu(in_ch, in_ch)

        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

        self.conv2P = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))
        self.conv2C = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))

        self.conv3 = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2P(self.PAM(x)) + self.conv2C(self.CAM(x))
        return self.conv3(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class MAResUNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, pretrained=True):
        super(MAResUNet, self).__init__()
        self.name = "MAResUNet"

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.attention4 = PAM_CAM_Layer(filters[3])
        self.attention3 = PAM_CAM_Layer(filters[2])
        self.attention2 = PAM_CAM_Layer(filters[1])
        self.attention1 = PAM_CAM_Layer(filters[0])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.attention4(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


def build_maresunet():
    """Tiny-input MAResUNet for tracing (pretrained=False to skip a weights download; the
    architecture -- ResNet34 encoder + PAM_CAM attention + decoder -- is unmodified)."""
    model = MAResUNet(num_channels=3, num_classes=5, pretrained=False)
    model.eval()
    return model


def example_input_maresunet():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("MAResUNet", build_maresunet, example_input_maresunet, 2021, "vendored-pytorch"),
]
