# SOURCE: vendored from https://github.com/teresasun/docUnet.pytorch @ master
# (models/doc_unet/unet_parts.py, models/doc_unet/model.py)
#
# DocUNet (Ma et al., CVPR 2018, "DocUNet: Document Image Unwarping via A
# Stacked U-Net"): a stacked two-stage U-Net that predicts a forward mapping
# to unwarp a photographed/scanned document image. This is the real
# `Doc_UNet` / `UNet` model code from the docUnet.pytorch repository -- the
# first `UNet` stage returns both its output map and its penultimate feature
# maps, which are concatenated with stage-1's output and fed into a second
# `UNet` stage. No architecture was altered; only the `from
# models.doc_unet.unet_parts import *` relative import was inlined into a
# single file for menagerie staging.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/doc_unet/unet_parts.py (real repo code, inlined)
# ---------------------------------------------------------------------------


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2, output feature map size unchanged."""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# models/doc_unet/model.py (real repo code, inlined)
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, need_feature_maps=False):
        super(UNet, self).__init__()
        self.need_feature_maps = need_feature_maps
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y = self.outc(x)
        if self.need_feature_maps:
            return y, x
        return y


class Doc_UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Doc_UNet, self).__init__()
        self.U_net1 = UNet(input_channels, n_classes, need_feature_maps=True)
        self.U_net2 = UNet(64 + n_classes, n_classes, need_feature_maps=False)

    def forward(self, x):
        y1, feature_maps = self.U_net1(x)
        x = torch.cat((feature_maps, y1), dim=1)
        y2 = self.U_net2(x)
        return y1, y2


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_INPUT_CHANNELS = 3
_N_CLASSES = 2  # DocUNet predicts a 2-channel (x, y) forward mapping
_SPATIAL = 64  # divisible by 16 (4 down/up stages) to keep U-Net skip shapes exact


def build_docunet():
    torch.manual_seed(0)
    model = Doc_UNet(_INPUT_CHANNELS, _N_CLASSES)
    model.eval()
    return model


def example_input_docunet():
    torch.manual_seed(0)
    return torch.randn(1, _INPUT_CHANNELS, _SPATIAL, _SPATIAL)


MENAGERIE_ENTRIES = [
    ("DocUNet", "build_docunet", "example_input_docunet", 2018, MENAGERIE_ZOO),
]
