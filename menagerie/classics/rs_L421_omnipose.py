# SOURCE: vendored from Makelalab/Omnipose @ main
# https://raw.githubusercontent.com/Makelalab/Omnipose/main/cellpose_omni/resnet_torch.py
#
# Cutler, Stringer, Lo, Rappez, Stroustrup, Brook Peterson, Wiggins, Mougous 2022
# (Nature Methods) "Omnipose: a high-precision morphology-independent solution for
# bacterial cell segmentation" -- a Cellpose-derived style-conditioned residual U-Net
# (`CPnet`) that predicts per-pixel flow fields (+ cell-probability/boundary classes)
# for morphology-independent instance segmentation, including elongated/filamentous
# bacterial cells that break the "roundish cell" assumption of the original Cellpose.
# `dilation_list`, `batchconv`, `resdown`, `convdown`, `resup`, `convup`,
# `batchconvstyle`, `downsample`, `upsample`, `make_style`, `CPnet` are copied verbatim
# from the real `cellpose_omni/resnet_torch.py`. The only change is dropping the
# `from omnipose.gpu import ARM, torch_GPU, torch_CPU, empty_cache` import and the
# `save_model`/`load_model` checkpoint-I/O methods that depended on it (mechanical
# import-trim; not exercised by tracing a forward pass, no architectural change).
#
# Real construction defaults (from `cellpose_omni/core.py` `UnetModel.__init__`,
# `nsample=4`, `nchan=1`, omni 2D `nclasses=2+(dim-1)=3`): `nbase=[1,32,64,128,256]`,
# `nout=3`, `sz=3`, `residual_on=True`, `style_on=True`, `concatenation=False`,
# `dim=2`, `kernel_size=2`, `scale_factor=2`, `dilation=1`.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def dilation_list(x, N):
    return np.round(np.linspace(1, x, N)).astype(int).tolist()


def batchconv(in_channels, out_channels, kernel_size, dim, dilation, relu=True):
    BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    ConvND = nn.Conv2d if dim == 2 else nn.Conv3d

    # Adjust padding for dilated convolutions
    padding = ((kernel_size - 1) * dilation) // 2

    layers = [BatchNorm(in_channels, eps=1e-5, momentum=0.05)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers.append(
        ConvND(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
        )
    )

    return nn.Sequential(*layers)


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        super().__init__()

        self.conv = nn.Sequential()
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)

        for t in range(4):
            if t == 0:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(in_channels, out_channels, sz, dim, dilation)
                )
            else:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(out_channels, out_channels, sz, dim, dilation)
                )

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(in_channels, out_channels, sz, dim, dilation)
                )
            else:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(out_channels, out_channels, sz, dim, dilation)
                )

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation

        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels,
                out_channels,
                style_channels,
                sz,
                dim,
                dilation,
                concatenation=concatenation,
            ),
        )
        self.conv.add_module(
            "conv_2", batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation)
        )
        self.conv.add_module(
            "conv_3", batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation)
        )
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x


class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels,
                out_channels,
                style_channels,
                sz,
                dim,
                dilation,
                concatenation=concatenation,
            ),
        )

    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x) + y)
        return x


class batchconvstyle(nn.Module):
    def __init__(
        self, in_channels, out_channels, style_channels, sz, dim, dilation, concatenation=False
    ):
        super().__init__()
        self.conv = batchconv(in_channels, out_channels, sz, dim, dilation)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.full = nn.Linear(style_channels, out_channels)
        self.dim = dim

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            x = x + y

        feat = self.full(style)

        for k in range(self.dim):
            feat = feat.unsqueeze(-1)

        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y


class downsample(nn.Module):
    def __init__(self, parent):
        super().__init__()
        nbase = parent.nbase
        sz = parent.sz
        residual_on = parent.residual_on
        dim = parent.dim
        dilation = parent.dilation
        kernel_size = parent.kernel_size
        scale_factor = parent.scale_factor

        self.checkpoint = parent.checkpoint
        self.down = nn.Sequential()

        maxpool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d

        self.maxpool = maxpool(kernel_size=kernel_size, stride=scale_factor)

        N = len(nbase) - 1
        dilations = dilation_list(dilation, N)
        for n, dilation in enumerate(dilations):
            if residual_on:
                self.down.add_module(
                    "res_down_%d" % n, resdown(nbase[n], nbase[n + 1], sz, dim, dilation)
                )
            else:
                self.down.add_module(
                    "conv_down_%d" % n, convdown(nbase[n], nbase[n + 1], sz, dim, dilation)
                )

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = (
                    cp.checkpoint(self.maxpool, xd[n - 1])
                    if self.checkpoint
                    else self.maxpool(xd[n - 1])
                )
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class upsample(nn.Module):
    def __init__(self, parent):
        super().__init__()
        nbase = parent.nbaseup
        kernel_size = parent.kernel_size  # noqa: F841 -- unused in real source too (verbatim)
        scale_factor = parent.scale_factor

        self.upsampling = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.up = nn.Sequential()
        self.checkpoint = parent.checkpoint

        N = len(nbase) - 1
        dilations = dilation_list(parent.dilation, N)
        for k, dilation in enumerate(dilations):
            n = k + 1
            if parent.residual_on:
                self.up.add_module(
                    "res_up_%d" % (n - 1),
                    resup(nbase[n], nbase[n - 1], nbase[-1], parent, dilation=dilation),
                )
            else:
                self.up.add_module(
                    "conv_up_%d" % (n - 1),
                    convup(nbase[n], nbase[n - 1], nbase[-1], parent, dilation=dilation),
                )

    def forward(self, style, xd, mkldnn=False):
        x = xd[-1]
        for n in range(len(self.up)):
            idx = -(n + 1)
            if n > 0:
                if mkldnn:
                    x = self.upsampling(x.to_dense()).to_mkldnn()
                else:
                    x = cp.checkpoint(self.upsampling, x) if self.checkpoint else self.upsampling(x)

            x = (
                cp.checkpoint(self.up[idx], x, xd[idx], style, mkldnn)
                if self.checkpoint
                else self.up[idx](x, xd[idx], style, mkldnn=mkldnn)
            )

        return x


class make_style(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.dim = parent.dim
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool2d if self.dim == 2 else F.avg_pool3d

    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=tuple(x0.shape[-self.dim :]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5

        return style


class CPnet(nn.Module):
    def __init__(
        self,
        nbase,
        nout,
        sz,
        residual_on=True,
        style_on=True,
        concatenation=False,
        mkldnn=False,
        dim=2,
        checkpoint=False,
        dropout=False,
        kernel_size=2,
        scale_factor=2,
        dilation=1,
    ):
        super(CPnet, self).__init__()

        self.checkpoint = checkpoint
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.dim = dim
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(self)

        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.nbaseup = nbaseup

        self.upsample = upsample(self)
        self.make_style = make_style(self)
        self.output = batchconv(nbaseup[0], nout, 1, self.dim, 1)

        self.style_on = style_on

        self.do_dropout = dropout
        if self.do_dropout:
            self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)

        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
            style = (
                cp.checkpoint(self.make_style, T0[-1])
                if self.checkpoint
                else self.make_style(T0[-1])
            )

        style0 = style
        if not self.style_on:
            style = style * 0

        T0 = self.upsample(style, T0, self.mkldnn)

        if self.do_dropout:
            T0 = self.dropout(T0)

        T0 = cp.checkpoint(self.output, T0) if self.checkpoint else self.output(T0)

        if self.mkldnn:
            T0 = T0.to_dense()

        return T0, style0


def build_omnipose():
    # nsample=4, nchan=1 real defaults -> nbase=[1,32,64,128,256]; omni 2D default
    # nclasses = 2+(dim-1) = 3 (flow-x, flow-y, cell-probability).
    nbase = [1, 32, 64, 128, 256]
    return CPnet(
        nbase,
        nout=3,
        sz=3,
        residual_on=True,
        style_on=True,
        concatenation=False,
        dim=2,
        kernel_size=2,
        scale_factor=2,
        dilation=1,
    )


def example_input_omnipose():
    # (batch, 1-channel grayscale microscopy image, H, W); 64x64 keeps the 4-level
    # downsample/upsample U-Net tiny while dividing evenly by scale_factor**4=16.
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Omnipose", "build_omnipose", "example_input_omnipose", 2022, "vendored"),
]
