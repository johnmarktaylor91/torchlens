# SOURCE: vendored from hellopipu/HQS-Net @ main
# Files: model/BasicModule.py, model/HQSNet.py
# https://github.com/hellopipu/HQS-Net
#
# HQS-Net: "Learned Half-Quadratic Splitting Network for MR Image Reconstruction"
# (Xin et al., MIDL/OpenReview https://openreview.net/pdf?id=h7rXUbALijU). Unrolled-
# optimization MRI reconstruction network: n_iter reconstruction blocks (regular CNN
# blocks by default) interleaved with a k-space-consistency HQS update step that uses
# torch.fft.fft2/ifft2 for the (under-sampled) forward/backward MRI operator.
#
# Import-fix only (per rung-2 rules, architecture code is untouched): the two source
# files used package-relative imports (`from model.BasicModule import conv_block`,
# `from model.BasicModule import UNetRes`) across separate files in the original repo;
# concatenated here into one staging module, those become plain in-file references. No
# other code was changed. UNetRes (the alternate 'unet' block_type) is vendored too but
# unused by the default entry point below (block_type='cnn' matches the paper's main
# reported configuration).

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- model/BasicModule.py (verbatim) ----------------------------------------------------


def conv_block(model_name="hqs-net", channel_in=22, n_convs=3, n_filters=32):
    """
    reconstruction blocks in DC-CNN;
    primal(image)-net blocks and dual(k)-space-net blocks in LPD-Net;
    regular cnn reconstruction blocks in HQS-Net
    :param model_name: 'dc-cnn', 'prim-net', 'dual-net', or 'hqs-net'
    :param channel_in:
    :param n_filters:
    :param n_convs:
    :return:
    """
    layers = []
    if model_name == "dc-cnn":
        channel_out = channel_in
    elif model_name == "prim-net" or model_name == "hqs-net":
        channel_out = channel_in - 2
    elif model_name == "dual-net":
        channel_out = channel_in - 4

    for i in range(n_convs - 1):
        if i == 0:
            layers.append(nn.Conv2d(channel_in, n_filters, kernel_size=3, stride=1, padding=1))
        else:
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(nn.Conv2d(n_filters, channel_out, kernel_size=3, stride=1, padding=1))

    return nn.Sequential(*layers)


"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


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


"""
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# resblock (ResBlock)
# --------------------------------------------
"""


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


def upsample_pixelshuffle(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
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
        negative_slope=negative_slope,
    )
    return up1


def upsample_upconv(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR"
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode,
        negative_slope=negative_slope,
    )
    return up1


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


def downsample_maxpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


def downsample_avgpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


## UnetRes is taken from https://github.com/cszn/DPIR/blob/master/models/network_unet.py
## used as modified Unet reconstruction blocks in HQS-Net
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


# --- model/HQSNet.py (verbatim) -----------------------------------------------------------


class HQSNet(nn.Module):
    def __init__(
        self, buffer_size=5, n_iter=8, n_convs=6, n_filters=64, block_type="cnn", norm="ortho"
    ):
        """
        HQS-Net from paper " Learned Half-Quadratic Splitting Network for MR Image Reconstruction "
        ( https://openreview.net/pdf?id=h7rXUbALijU ) ( https://github.com/hellopipu/HQS-Net )
        :param buffer_size:  buffer_size m
        :param n_iter:  iterations n
        :param n_convs: convolutions in each reconstruction block
        :param n_filters: output channel for convolutions
        :param block_type: 'cnn' or 'unet
        :param norm: 'ortho' norm for fft
        """

        super().__init__()
        self.norm = norm
        self.m = buffer_size
        self.n_iter = n_iter
        ## the initialization of mu may influence the final accuracy
        self.mu = nn.Parameter(0.5 * torch.ones((1, 1)))  # 2
        self.block_type = block_type
        if self.block_type == "cnn":
            rec_blocks = []
            for i in range(self.n_iter):
                rec_blocks.append(
                    conv_block(
                        "hqs-net", channel_in=2 * (self.m + 1), n_convs=n_convs, n_filters=n_filters
                    )
                )  # self.m +
            self.rec_blocks = nn.ModuleList(rec_blocks)
        elif self.block_type == "unet":
            self.rec_blocks = UNetRes(
                in_nc=2 * (self.m + 1),
                out_nc=2 * self.m,
                nc=[64, 128, 256, 512],
                nb=4,
                act_mode="R",
                downsample_mode="strideconv",
                upsample_mode="convtranspose",
            )

    def _forward_operation(self, img, mask):
        k = torch.fft.fft2(
            torch.view_as_complex(img.permute(0, 2, 3, 1).contiguous()), norm=self.norm
        )
        k = torch.view_as_real(k).permute(0, 3, 1, 2).contiguous()
        k = mask * k
        return k

    def _backward_operation(self, k, mask):
        k = mask * k
        img = torch.fft.ifft2(
            torch.view_as_complex(k.permute(0, 2, 3, 1).contiguous()), norm=self.norm
        )
        img = torch.view_as_real(img).permute(0, 3, 1, 2).contiguous()
        return img

    def update_opration(self, f_1, k, mask):
        h_1 = k - self._forward_operation(f_1, mask)
        update = f_1 + self.mu * self._backward_operation(h_1, mask)
        return update

    def forward(self, img, k, mask):
        """
        :param img: zero-filled images, (batch,2,h,w)
        :param k:   corresponding undersampled k-space data , (batch,2,h,w)
        :param mask: uncentered sampling mask , (batch,2,h,w)
        :return: reconstructed img
        """

        ## initialize buffer f : the concatenation of m copies of the complex-valued zero-filled images
        f = torch.cat([img] * self.m, 1).to(img.device)

        ## n reconstruction blocks
        for i in range(self.n_iter):
            f_1 = f[:, 0:2].clone()
            updated_f_1 = self.update_opration(f_1, k, mask)
            if self.block_type == "cnn":
                f = f + self.rec_blocks[i](torch.cat([f, updated_f_1], 1))
            elif self.block_type == "unet":
                f = f + self.rec_blocks(torch.cat([f, updated_f_1], 1))
        return f[:, 0:2]


# --- staging entry points ----------------------------------------------------------------


def build_hqsnet():
    """Tiny random-init HQS-Net (unrolled half-quadratic-splitting MRI reconstruction)."""
    return HQSNet(buffer_size=2, n_iter=2, n_convs=3, n_filters=8, block_type="cnn", norm="ortho")


def example_input_hqsnet():
    """Real multi-tensor input: (zero-filled image, undersampled k-space, sampling mask).

    All three tensors share shape (batch, 2, h, w) -- channel dim 2 holds the real/imag
    parts of the complex-valued MRI data, matching the model's forward() docstring.
    """
    torch.manual_seed(0)
    batch, h, w = 1, 32, 32
    img = torch.randn(batch, 2, h, w)
    k = torch.randn(batch, 2, h, w)
    mask = torch.zeros(batch, 2, h, w)
    mask[:, :, :, ::4] = 1.0  # simple undersampling pattern
    return (img, k, mask)


MENAGERIE_ENTRIES = [
    (
        "HQS-Net",
        "build_hqsnet",
        "example_input_hqsnet",
        "2021",
        "SOURCE_AVAILABLE",
    ),
]
