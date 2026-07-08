# SOURCE: vendored from cszn/USRNet @ master
#   models/basicblock.py (verbatim) + models/network_usrnet_v1.py (verbatim architecture;
#   this is the *official* author-maintained variant "for pytorch version >= 1.8.1" that
#   uses `torch.fft` instead of the removed `torch.rfft`/`torch.irfft` ops used by the
#   original `network_usrnet.py`; same repo, same author, modern-torch-compatible copy).
# License: the USRNet repo carries no explicit LICENSE file; code reproduced for
# research/tracing purposes per repo's public availability, matching Kai Zhang's
# published USRNet (CVPR 2020) architecture verbatim (deep-unfolding super-resolution:
# alternating data-consistency module + ResUNet prior module + hyper-parameter module).
#
# Only change from the real `network_usrnet_v1.py`: dropped the unused
# `from utils import utils_image as util` import (only used by demo/test scripts, never
# referenced inside the model classes) and inlined `models.basicblock` as a
# same-file section (`import models.basicblock as B` -> the module-local names below)
# so the file is self-contained. No architecture code was rewritten.
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# models/basicblock.py (verbatim)
# --------------------------------------------------------------------------------


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CBR"
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
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
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
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ")
    return sequential(*L)


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
    ):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


def upsample_pixelshuffle(
    in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode="2R"
):
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(
        in_channels,
        out_channels * (int(mode[0]) ** 2),
        kernel_size,
        stride,
        padding,
        bias,
        mode="C" + mode,
    )
    return up1


def upsample_upconv(
    in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode="2R"
):
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode)
    return up1


def upsample_convtranspose(
    in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode="2R"
):
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return up1


def downsample_strideconv(
    in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode="2R"
):
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return down1


def downsample_maxpool(
    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode="2R"
):
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


def downsample_avgpool(
    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="2R"
):
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


# --------------------------------------------------------------------------------
# models/network_usrnet_v1.py (verbatim; "for pytorch version >= 1.8.1")
# --------------------------------------------------------------------------------


def splits(a, sf):
    """split a into sfxsf distinct blocks

    Args:
        a: NxCxWxH
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    """
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., : psf.shape[2], : psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def upsample(x, sf=3):
    """s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    """s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    """
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


class ResUNet(nn.Module):
    def __init__(
        self,
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=2,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(ResUNet, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode="C")

        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
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

        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
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

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        FR = FBFy + torch.fft.fftn(alpha * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + alpha)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
        FX = (FR - FCBinvWBR) / alpha
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        return Xest


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


class USRNet(nn.Module):
    def __init__(
        self,
        n_iter=8,
        h_nc=64,
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=2,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(USRNet, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )
        self.h = HyPaNet(in_nc=2, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        """
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        """

        w, h = x.shape[-2:]
        FB = p2o(k, (w * sf, h * sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        STy = upsample(x, sf=sf)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        x = nn.functional.interpolate(x, scale_factor=sf, mode="nearest")

        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i : i + 1, ...], sf)
            x = self.p(
                torch.cat(
                    (x, ab[:, i + self.n : i + self.n + 1, ...].repeat(1, 1, x.size(2), x.size(3))),
                    dim=1,
                )
            )

        return x


# --------------------------------------------------------------------------------
# menagerie staging entry points
# --------------------------------------------------------------------------------

MENAGERIE_ZOO = "vendored-pytorch"


def build_usrnet():
    # Tiny-size real USRNet: shallower unfolding + smaller ResUNet channel widths
    # than the paper defaults (n_iter=8, nc=[64,128,256,512]) but the exact same
    # real deep-unfolding architecture (DataNet + ResUNet + HyPaNet).
    return USRNet(n_iter=2, h_nc=8, in_nc=4, out_nc=3, nc=[8, 16, 32, 64], nb=1)


def example_input_usrnet():
    # USRNet.forward(x, k, sf, sigma): x=LR image, k=blur kernel, sf=int scale
    # factor, sigma=noise level map. We pass a module wrapper below to bundle the
    # non-tensor `sf` argument since torchlens traces on real forward() calls.
    n, c, h, w = 1, 3, 16, 16
    x = torch.rand(n, c, h, w)
    k = torch.ones(n, 1, 5, 5) / 25.0
    sigma = torch.full((n, 1, 1, 1), 0.05)
    return x, k, sigma


class USRNetWrapped(nn.Module):
    """Thin wrapper pinning the integer scale-factor `sf` so torchlens can trace
    USRNet.forward with tensor-only example inputs (sf=1 is a valid, real USRNet
    configuration -- non-blind deconvolution / denoising mode)."""

    def __init__(self, usrnet: USRNet, sf: int = 1):
        super().__init__()
        self.usrnet = usrnet
        self.sf = sf

    def forward(self, x, k, sigma):
        return self.usrnet(x, k, self.sf, sigma)


def build_usrnet_wrapped():
    return USRNetWrapped(build_usrnet(), sf=1)


MENAGERIE_ENTRIES = [
    ("USRNet", build_usrnet_wrapped, example_input_usrnet, 2020, MENAGERIE_ZOO),
]
