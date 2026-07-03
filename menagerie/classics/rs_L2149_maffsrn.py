# SOURCE: vendored from AbdulMoqeet/MAFFSRN @ master
# https://raw.githubusercontent.com/AbdulMoqeet/MAFFSRN/master/MAFFSRN/src/model/maffsrn.py
#
# Muqeet, Hwang, Yang, Kang, Kim, Bae 2020 "Multi-attention based Ultra Lightweight Image
# Super-Resolution" (AIM 2020 workshop, ECCV; AIM 2020 efficient super-resolution challenge
# winner) -- feature-fusion groups (FFG) of multi-attention blocks (MAB, combining spatial
# pooling/upsample gating with two dilated conv branches and channel-shuffle feature fusion),
# a weight-normalized head/body/tail design, and a dual-kernel (3x3 + 5x5) pixel-shuffle tail
# with learnable per-branch scale. Scale/Tail/pixel_unshuffle/channel_shuffle/FFG/MAB/MAFFSRN
# and conv_block/activation/pad/sequential/get_valid_padding are copied verbatim from
# model/maffsrn.py; only the `make_model(args)` factory (which is generic ARGS-driven
# infrastructure from the shared EDSR-style `src/model/__init__.py` harness, not part of the
# architecture) is replaced by a direct `build_maffsrn()` that constructs a small
# `argparse.Namespace` with the repo's documented defaults (n_feats=32, n_FFGs=8, scale=[2],
# n_colors=3) instead of parsing CLI args.
"""MAFFSRN: multi-attention feature-fusion groups for ultra-lightweight super-resolution
(Muqeet et al. 2020, AIM 2020 SR challenge winner)."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = activation(act_type) if act_type else None
    # NOTE: `norm` is never defined upstream either -- this branch is dead code in the
    # original repo because every conv_block call site in maffsrn.py passes norm_type=None.
    n = norm(norm_type, out_nc) if norm_type else None  # noqa: F821
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("activation layer [{:s}] is not found".format(act_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError("padding layer [%s] is not implemented" % pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Tail(nn.Module):
    def __init__(self, args, scale, n_feats, kernel_size, wn):
        super(Tail, self).__init__()
        out_feats = scale * scale * args.n_colors
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))

        return x0 + x1


def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]

    kernel = torch.zeros(
        size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
        device=input.device,
    )
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor :: downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class FFG(nn.Module):
    def __init__(self, n_feats, wn, act=nn.ReLU(True)):
        super(FFG, self).__init__()

        self.b0 = MAB(n_feats=n_feats, reduction_factor=4)
        self.b1 = MAB(n_feats=n_feats, reduction_factor=4)
        self.b2 = MAB(n_feats=n_feats, reduction_factor=4)
        self.b3 = MAB(n_feats=n_feats, reduction_factor=4)

        self.reduction1 = wn(nn.Conv2d(n_feats * 2, n_feats, 1))
        self.reduction2 = wn(nn.Conv2d(n_feats * 2, n_feats, 1))
        self.reduction3 = wn(nn.Conv2d(n_feats * 2, n_feats, 1))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0) + x0
        x2 = self.b2(x1) + x1
        x3 = self.b3(x2)

        res1 = self.reduction1(channel_shuffle(torch.cat([x0, x1], dim=1), 2))
        res2 = self.reduction2(channel_shuffle(torch.cat([res1, x2], dim=1), 2))
        res = self.reduction3(channel_shuffle(torch.cat([res2, x3], dim=1), 2))

        return self.res_scale(res) + self.x_scale(x)


class MAB(nn.Module):
    def __init__(self, n_feats, reduction_factor=4, distillation_rate=0.25):
        super(MAB, self).__init__()
        self.reduce_channels = nn.Conv2d(n_feats, n_feats // reduction_factor, 1)
        self.reduce_spatial_size = nn.Conv2d(
            n_feats // reduction_factor, n_feats // reduction_factor, 3, stride=2, padding=1
        )
        self.pool = nn.MaxPool2d(7, stride=3)
        self.increase_channels = conv_block(n_feats // reduction_factor, n_feats, 1)

        self.conv1 = conv_block(
            n_feats // reduction_factor,
            n_feats // reduction_factor,
            3,
            dilation=1,
            act_type="lrelu",
        )
        self.conv2 = conv_block(
            n_feats // reduction_factor,
            n_feats // reduction_factor,
            3,
            dilation=2,
            act_type="lrelu",
        )

        self.sigmoid = nn.Sigmoid()

        self.conv00 = conv_block(n_feats, n_feats, 3, act_type=None)
        self.conv01 = conv_block(n_feats, n_feats, 3, act_type="lrelu")

        self.bottom11 = conv_block(n_feats, n_feats, 1, act_type=None)
        self.bottom11_dw = conv_block(n_feats, n_feats, 5, groups=n_feats, act_type=None)

    def forward(self, x):
        x = self.conv00(self.conv01(x))
        rc = self.reduce_channels(x)
        rs = self.reduce_spatial_size(rc)
        pool = self.pool(rs)
        conv = self.conv2(pool)
        conv = conv + self.conv1(pool)
        up = torch.nn.functional.interpolate(conv, size=(rc.shape[2], rc.shape[3]), mode="nearest")
        up = up + rc
        out = (self.sigmoid(self.increase_channels(up)) * x) * self.sigmoid(
            self.bottom11_dw(self.bottom11(x))
        )
        return out


class MAFFSRN(nn.Module):
    def __init__(self, args):
        super(MAFFSRN, self).__init__()

        # hyper-params
        self.scale = args.scale
        n_FFGs = args.n_FFGs
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.LeakyReLU(True)

        wn = lambda x: torch.nn.utils.weight_norm(x)  # noqa: E731

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([0.4488, 0.4371, 0.4040])).view(
            [1, 3, 1, 1]
        )

        # define head module
        head = []
        head.append(wn(nn.Conv2d(3, n_feats, 3, padding=3 // 2)))

        # define body module
        body = []
        for i in range(n_FFGs):
            body.append(FFG(n_feats, wn=wn, act=act))

        # define tail module
        tail = Tail(args, self.scale[0], n_feats, kernel_size, wn)

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail

    def forward(self, x):
        input = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + torch.nn.functional.interpolate(
            input, scale_factor=self.scale[0], mode="bicubic"
        )


def build_maffsrn():
    # Repo defaults from MAFFSRN/src/option.py: scale='2' -> [2], n_colors=3,
    # n_feats=32, n_FFGs=8 (shrunk here to keep the traced graph small).
    args = argparse.Namespace(scale=[2], n_colors=3, n_feats=8, n_FFGs=1)
    return MAFFSRN(args)


def example_input_maffsrn():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("MAFFSRN", "build_maffsrn", "example_input_maffsrn", 2020, "vendored"),
]
