# SOURCE: vendored from cszn/DPIR @ 15bca3fcc1f3cc51a1f99ccf027691e278c19354
# https://raw.githubusercontent.com/cszn/DPIR/15bca3fcc1f3cc51a1f99ccf027691e278c19354/models/network_unet.py
# https://raw.githubusercontent.com/cszn/DPIR/15bca3fcc1f3cc51a1f99ccf027691e278c19354/models/basicblock.py
#
# Kai Zhang et al. 2021 "Plug-and-Play Image Restoration with Deep Denoiser
# Prior" (IEEE TPAMI) -- official PyTorch implementation. DRUNet is the
# `UNetRes` class: a residual U-Net denoiser that additionally takes a
# per-pixel noise-level map concatenated onto the input channels, four-scale
# strided-conv down / conv-transpose up encoder-decoder with residual blocks
# at every scale and long-range additive skip connections (`x + xN` rather
# than concatenation) -- the network used as the "deep denoiser prior" plugged
# into the paper's HQS/PnP restoration algorithms (deblurring, super-resolution,
# demosaicking).
#
# `UNetRes` (from `models/network_unet.py`) is copied verbatim, along with the
# exact subset of `models/basicblock.py` it calls at its default
# `downsample_mode='strideconv'` / `upsample_mode='convtranspose'` settings:
# `sequential`, `conv`, `ResBlock`, `downsample_strideconv`,
# `upsample_convtranspose`. No architectural changes were made; only
# mechanical fixes for import isolation:
#   - `import models.basicblock as B` -> the needed `B.*` helpers are defined
#     directly in this module (verbatim bodies, copied from basicblock.py)
#     and `UNetRes` calls them unqualified.
# All imports (`torch`, `torch.nn`) are base libs already installed.

from collections import OrderedDict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# models/basicblock.py (verbatim subset used by UNetRes at its default
# downsample_mode='strideconv' / upsample_mode='convtranspose' settings)
# ---------------------------------------------------------------------------


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "2":
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "3":
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == "4":
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == "U":
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ")
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        mode="CRC",
        negative_slope=0.2,
    ):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(
            in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope
        )

    def forward(self, x):
        return x + self.res(x)


# --------------------------------------------
# strideconv + relu
# --------------------------------------------
def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope
    )
    return down1


# --------------------------------------------
# convTranspose + relu
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


# ---------------------------------------------------------------------------
# models/network_unet.py: UNetRes (this is DRUNet)
# ---------------------------------------------------------------------------


class UNetRes(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(UNetRes, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError("downsample mode [{:s}] is not found".format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode="2"),
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode="2"),
        )
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode="2"),
        )

        self.m_body = sequential(
            *[ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        # upsample
        if upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError("upsample mode [{:s}] is not found".format(upsample_mode))

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
        )

        self.m_tail = conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x


def build_drunet():
    # in_nc=4 (RGB + noise-level map, the paper's actual denoiser-prior input),
    # out_nc=3, shrunk channel widths / block counts for a fast smoke trace.
    return UNetRes(in_nc=4, out_nc=3, nc=[16, 32, 64, 128], nb=2)


def example_input_drunet():
    return (torch.randn(1, 4, 32, 32),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DRUNet", "build_drunet", "example_input_drunet", 2021, "vendored"),
]
