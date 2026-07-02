# SOURCE: vendored from edongdongchen/EI @ master
# https://github.com/edongdongchen/EI
# File: models/unet.py -- the generator/reconstruction network trained by the
# Equivariant Imaging (ICCV 2021 Oral, arXiv:2103.14756) methodology and its
# REI variant (CVPR 2022). Equivariant Imaging is a *training* scheme (an
# equivariance loss over a transformation group) applied to this residual
# circular-padded UNet reconstruction network; the network architecture below
# is transcribed verbatim from the official repo (only the class name/module
# path context changed to be self-contained).
#
# NOTE: this single vendored module also covers the catalog's separate
# "Equivariant Imaging Network" entry -- both point at the same upstream repo
# and the same models/unet.py generator (catalog-flagged POTENTIAL_DEDUP of
# each other), so only one staging build is emitted for the pair.
import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, circular_padding=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                padding_mode="circular" if circular_padding else "zeros",
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """Equivariant Imaging reconstruction network (residual UNet generator).

    Used as the generator `G` trained via the equivariance loss in
    edongdongchen/EI's `ei.ei.EI.train_ei` (imaging inverse problems: CT,
    inpainting, etc). `compact=4` (the repo's default / paper config) skips
    the deepest encoder/decoder stage relative to a standard 5-stage UNet.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        compact=4,
        residual=True,
        circular_padding=False,
        cat=True,
    ):
        super(UNet, self).__init__()
        self.name = "unet"
        self.residual = residual
        self.cat = cat

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64, circular_padding=circular_padding)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

        if compact == 5:
            self._forward = self.forward_standard
        if compact == 4:
            self._forward = self.forward_compact4

    def forward(self, x):
        return self._forward(x)

    def forward_standard(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        if self.cat:
            d5 = torch.cat((x4, d5), dim=cat_dim)
            d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual else d1
        return out

    def forward_compact4(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual else d1
        return out


def build_ei_unet():
    # Matches ei.ei.EI.train_ei's generator construction:
    # UNet(in_channels, out_channels, compact=4, residual=residual,
    #      circular_padding=True, cat=cat)
    return UNet(
        in_channels=3, out_channels=3, compact=4, residual=True, circular_padding=True, cat=True
    )


def example_input_ei_unet():
    # get_started.py's inpainting demo uses 256x256 RGB Urban100 crops;
    # a small even-power-of-two spatial size is used here for tracing so the
    # 3 Maxpool/up_conv stages (compact=4) divide evenly.
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Equivariant Imaging", "build_ei_unet", "example_input_ei_unet", 2021, MENAGERIE_ZOO),
]
