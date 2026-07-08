# SOURCE: vendored from czbiohub-sf/noise2self @ master (models/unet.py, models/babyunet.py,
# models/dncnn.py, models/singleconv.py, models/modules.py)
"""Noise2Self (ICML 2019) J-invariant blind-spot denoising models.

The paper's official PyTorch repo (czbiohub-sf/noise2self) ships four interchangeable
denoiser backbones selectable via models/models.py:get_model(): a full multi-scale U-Net,
a "BabyUnet" (2-level shallow variant), DnCNN, and a single-conv baseline. This file
vendors all four nn.Module definitions verbatim (only import paths adjusted to be
self-contained -- no relative "models.modules" package). The blind-spot J-invariant
masking (mask.py) is a data-augmentation/training-loop concern, not part of the traced
forward architecture, so it is not vendored here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/modules.py (verbatim, shared ConvBlock used by BabyUnet and Unet)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False,
        norm="batch",
        residual=True,
        activation="leakyrelu",
        transpose=False,
    ):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == "batch":
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == "instance":
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "mixed":
            self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == "relu":
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == "leakyrelu":
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == "elu":
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == "selu":
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        if self.norm1:
            x = self.norm1(x)

        x = self.actfun1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        if self.residual:
            x[:, 0 : min(ox.shape[1], x.shape[1]), :, :] += ox[
                :, 0 : min(ox.shape[1], x.shape[1]), :, :
            ]

        x = self.actfun2(x)

        return x


# ---------------------------------------------------------------------------
# models/babyunet.py (verbatim)
# ---------------------------------------------------------------------------
class BabyUnet(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, width=16):
        super(BabyUnet, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.up1 = lambda x: F.interpolate(x, mode="bilinear", scale_factor=2, align_corners=False)
        self.up2 = lambda x: F.interpolate(x, mode="bilinear", scale_factor=2, align_corners=False)

        self.conv1 = ConvBlock(n_channel_in, width)
        self.conv2 = ConvBlock(width, 2 * width)

        self.conv3 = ConvBlock(2 * width, 2 * width)

        self.conv4 = ConvBlock(4 * width, 2 * width)
        self.conv5 = ConvBlock(3 * width, width)

        self.conv6 = nn.Conv2d(width, n_channel_out, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        x = self.pool1(c1)
        c2 = self.conv2(x)
        x = self.pool2(c2)
        x = self.conv3(x)

        x = self.up1(x)
        x = torch.cat([x, c2], 1)
        x = self.conv4(x)
        x = self.up2(x)
        x = torch.cat([x, c1], 1)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


# ---------------------------------------------------------------------------
# models/unet.py (verbatim)
# ---------------------------------------------------------------------------
class Unet(nn.Module):
    def __init__(
        self,
        n_channel_in=1,
        n_channel_out=1,
        residual=False,
        down="conv",
        up="tconv",
        activation="selu",
    ):
        super(Unet, self).__init__()

        self.residual = residual

        if down == "maxpool":
            self.down1 = nn.MaxPool2d(kernel_size=2)
            self.down2 = nn.MaxPool2d(kernel_size=2)
            self.down3 = nn.MaxPool2d(kernel_size=2)
            self.down4 = nn.MaxPool2d(kernel_size=2)
        elif down == "avgpool":
            self.down1 = nn.AvgPool2d(kernel_size=2)
            self.down2 = nn.AvgPool2d(kernel_size=2)
            self.down3 = nn.AvgPool2d(kernel_size=2)
            self.down4 = nn.AvgPool2d(kernel_size=2)
        elif down == "conv":
            self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, groups=128)
            self.down4 = nn.Conv2d(256, 256, kernel_size=2, stride=2, groups=256)

            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        if up == "bilinear" or up == "nearest":
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
        elif up == "tconv":
            self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, groups=256)
            self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, groups=128)
            self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, groups=64)
            self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, groups=32)

            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock(n_channel_in, 32, residual, activation)
        self.conv2 = ConvBlock(32, 64, residual, activation)
        self.conv3 = ConvBlock(64, 128, residual, activation)
        self.conv4 = ConvBlock(128, 256, residual, activation)

        self.conv5 = ConvBlock(256, 256, residual, activation)

        self.conv6 = ConvBlock(2 * 256, 128, residual, activation)
        self.conv7 = ConvBlock(2 * 128, 64, residual, activation)
        self.conv8 = ConvBlock(2 * 64, 32, residual, activation)
        self.conv9 = ConvBlock(2 * 32, n_channel_out, residual, activation)

        if self.residual:
            self.convres = ConvBlock(n_channel_in, n_channel_out, residual, activation)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x, c4], 1)
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0))

        return x


# ---------------------------------------------------------------------------
# models/dncnn.py (verbatim)
# ---------------------------------------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


# ---------------------------------------------------------------------------
# models/singleconv.py (verbatim)
# ---------------------------------------------------------------------------
def pad_circular(x, pad):
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    if len(x.shape) == 2:
        x = torch.cat([x, x[0:pad]], dim=0)
        x = torch.cat([x, x[:, 0:pad]], dim=1)
        x = torch.cat([x[-2 * pad : -pad], x], dim=0)
        x = torch.cat([x[:, -2 * pad : -pad], x], dim=1)

    elif len(x.shape) == 4:
        x = torch.cat([x, x[:, :, 0:pad]], dim=2)
        x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
        x = torch.cat([x[:, :, -2 * pad : -pad], x], dim=2)
        x = torch.cat([x[:, :, :, -2 * pad : -pad], x], dim=3)

    return x


class SingleConvolution(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, width=3, torus=False):
        super(SingleConvolution, self).__init__()

        self.torus = torus

        if self.torus:
            self.pad = width // 2
            self.conv = nn.Conv2d(n_channel_in, n_channel_out, kernel_size=width, padding=0)
        else:
            self.conv = nn.Conv2d(
                n_channel_in, n_channel_out, kernel_size=width, padding=width // 2
            )

    def forward(self, x):
        if self.torus:
            x = pad_circular(x, self.pad)
            return self.conv(x)
        else:
            return self.conv(x)


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
def build_noise2self_unet():
    return Unet(n_channel_in=1, n_channel_out=1)


def example_input_noise2self_unet():
    return torch.randn(1, 1, 64, 64)


def build_noise2self_babyunet():
    return BabyUnet(n_channel_in=1, n_channel_out=1, width=8)


def example_input_noise2self_babyunet():
    return torch.randn(1, 1, 64, 64)


def build_noise2self_dncnn():
    return DnCNN(channels=1, num_of_layers=8)


def example_input_noise2self_dncnn():
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("Noise2Self-UNet", build_noise2self_unet, example_input_noise2self_unet, 2019, MENAGERIE_ZOO),
    (
        "Noise2Self-BabyUnet",
        build_noise2self_babyunet,
        example_input_noise2self_babyunet,
        2019,
        MENAGERIE_ZOO,
    ),
    (
        "Noise2Self-DnCNN",
        build_noise2self_dncnn,
        example_input_noise2self_dncnn,
        2019,
        MENAGERIE_ZOO,
    ),
]
