# SOURCE: vendored from https://github.com/hanyoseob/pytorch-noise2void @ master
# https://github.com/hanyoseob/pytorch-noise2void
#
# Files vendored (concatenated, imports/module-paths adjusted only):
#   layer.py  (CNR2d, DECNR2d, Conv2d, Deconv2d, Norm2d, ReLU, Pooling2d, UnPooling2d)
#   model.py  (UNet)
#
# The official Noise2Void implementation (juglab/n2v, referenced in the menagerie queue
# notes as the "official implementation") is TensorFlow/Keras-only
# (n2v/nets/unet.py imports `tensorflow.keras`), so it cannot be vendored into a torch
# staging module. Noise2Void's actual contribution is a *training* scheme (blind-spot
# masking of the self-supervised loss), not a novel architecture -- the network itself is
# a vanilla 2D U-Net. hanyoseob/pytorch-noise2void is a from-scratch PyTorch port that
# reimplements the same U-Net (avg-pool encoder, nearest-upsample decoder with skip
# concatenation, Conv-Norm-ReLU blocks) as juglab/n2v's TF U-Net, so we vendor its
# `UNet` class here unmodified as the real trainable architecture used for N2V-style
# blind-spot denoising. Only mechanical fixes applied: `from layer import *` collapsed
# into this single file (layer.py classes now co-resident), no architectural code changed.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# layer.py
# ---------------------------------------------------------------------------
class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class Deconv2d(nn.Module):
    def __init__(
        self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True
    ):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            nch_in,
            nch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        return self.deconv(x)


class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == "bnorm":
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == "inorm":
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Pooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type="avg"):
        super().__init__()

        if type == "avg":
            self.pooling = nn.AvgPool2d(pool)
        elif type == "max":
            self.pooling = nn.MaxPool2d(pool)
        elif type == "conv":
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type="nearest"):
        super().__init__()

        if type == "nearest":
            self.unpooling = nn.Upsample(scale_factor=pool, mode="nearest")
        elif type == "bilinear":
            self.unpooling = nn.Upsample(scale_factor=pool, mode="bilinear", align_corners=True)
        elif type == "conv":
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class CNR2d(nn.Module):
    def __init__(
        self,
        nch_in,
        nch_out,
        kernel_size=4,
        stride=1,
        padding=1,
        norm="bnorm",
        relu=0.0,
        drop=[],
        bias=[],
    ):
        super().__init__()

        if bias == []:
            if norm == "bnorm":
                bias = False
            else:
                bias = True

        layers = []
        layers += [
            Conv2d(
                nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        ]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class DECNR2d(nn.Module):
    def __init__(
        self,
        nch_in,
        nch_out,
        kernel_size=4,
        stride=1,
        padding=1,
        output_padding=0,
        norm="bnorm",
        relu=0.0,
        drop=[],
        bias=[],
    ):
        super().__init__()

        if bias == []:
            if norm == "bnorm":
                bias = False
            else:
                bias = True

        layers = []
        layers += [
            Deconv2d(
                nch_in,
                nch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            )
        ]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
# U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm="bnorm"):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == "bnorm":
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(
            1 * self.nch_in,
            1 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.enc1_2 = CNR2d(
            1 * self.nch_ker,
            1 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.pool1 = Pooling2d(pool=2, type="avg")

        self.enc2_1 = CNR2d(
            1 * self.nch_ker,
            2 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.enc2_2 = CNR2d(
            2 * self.nch_ker,
            2 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.pool2 = Pooling2d(pool=2, type="avg")

        self.enc3_1 = CNR2d(
            2 * self.nch_ker,
            4 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.enc3_2 = CNR2d(
            4 * self.nch_ker,
            4 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.pool3 = Pooling2d(pool=2, type="avg")

        self.enc4_1 = CNR2d(
            4 * self.nch_ker,
            8 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.enc4_2 = CNR2d(
            8 * self.nch_ker,
            8 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.pool4 = Pooling2d(pool=2, type="avg")

        self.enc5_1 = CNR2d(
            8 * self.nch_ker,
            2 * 8 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        """
        Decoder part
        """

        self.dec5_1 = DECNR2d(
            2 * 8 * self.nch_ker,
            8 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.unpool4 = UnPooling2d(pool=2, type="nearest")

        self.dec4_2 = DECNR2d(
            2 * 8 * self.nch_ker,
            8 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.dec4_1 = DECNR2d(
            8 * self.nch_ker,
            4 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.unpool3 = UnPooling2d(pool=2, type="nearest")

        self.dec3_2 = DECNR2d(
            2 * 4 * self.nch_ker,
            4 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.dec3_1 = DECNR2d(
            4 * self.nch_ker,
            2 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.unpool2 = UnPooling2d(pool=2, type="nearest")

        self.dec2_2 = DECNR2d(
            2 * 2 * self.nch_ker,
            2 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.dec2_1 = DECNR2d(
            2 * self.nch_ker,
            1 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )

        self.unpool1 = UnPooling2d(pool=2, type="nearest")

        self.dec1_2 = DECNR2d(
            2 * 1 * self.nch_ker,
            1 * self.nch_ker,
            kernel_size=3,
            stride=1,
            norm=self.norm,
            relu=0.0,
            drop=[],
        )
        self.dec1_1 = DECNR2d(
            1 * self.nch_ker,
            1 * self.nch_out,
            kernel_size=3,
            stride=1,
            norm=[],
            relu=[],
            drop=[],
            bias=False,
        )

    def forward(self, x):
        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Decoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = dec1

        return x


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
def build_n2v():
    # Tiny config: 1-channel grayscale in/out (typical microscopy denoising
    # setting), small base channel width for a fast trace.
    return UNet(nch_in=1, nch_out=1, nch_ker=8, norm="bnorm")


def example_input_n2v():
    # U-Net has 4 avg-pool downsamples (2^4=16); use a 64x64 input so all
    # skip-concat spatial sizes line up exactly.
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("N2V", "build_n2v", "example_input_n2v", 2019, "SOURCE_AVAILABLE"),
]
