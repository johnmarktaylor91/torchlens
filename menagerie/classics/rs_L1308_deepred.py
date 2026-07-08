# SOURCE: vendored from GaryMataev/DeepRED @ e772358832adbde681eabffb7fb8763eff3d2067
# (models/skip.py::skip; models/common.py::Interpolate, ::Concat, ::act, ::bn, ::conv;
#  models/downsampler.py::Downsampler, ::get_kernel, ::pad_circular;
#  utils/utils.py::fspecial_gauss; unchanged)
"""DeepRED: combining the Deep Image Prior (DIP) encoder-decoder ``skip`` network with
RED (Regularization by Denoising) as the image-restoration prior (Mataev, Milanfar, Elad,
"DeepRED: Deep Image Prior Powered by RED", ICCV Workshops 2019). Official repo:
https://github.com/GaryMataev/DeepRED (``models/skip.py`` @ master).

DeepRED's neural architecture *is* the Deep Image Prior ``skip`` network (Ulyanov,
Vedaldi, Lempitsky, "Deep Image Prior", CVPR 2018) -- an untrained convolutional
encoder-decoder with skip connections whose own randomly-initialized structure acts as an
implicit image prior; DeepRED's paper contribution is wrapping this network's per-pixel
reconstruction inside an outer ADMM optimization loop with the RED denoiser-based
regularizer (a training-loop/objective-level contribution, not an architectural one -- the
network itself, ``models/skip.py::skip()``, is used completely unmodified). This vendors
exactly that ``skip`` network builder plus its real, unmodified helper modules
(``Concat``, ``Interpolate``, ``act``, ``bn``, ``conv`` from ``common.py``, and
``Downsampler``/``get_kernel``/``pad_circular`` from ``downsampler.py`` for the
``downsample_mode='lanczos2'``/``'avg'``/``'max'`` code paths that ``conv()`` can reach --
though the default/documented config below uses ``downsample_mode='stride'``, which never
instantiates ``Downsampler``). No layer, channel count, connectivity, or forward-pass
control-flow was changed from the real repo.

The only non-architectural adaptation: the repo's ``models/skip.py`` does
``from .common import *`` and ``models/common.py`` does
``from .downsampler import Downsampler`` and ``models/downsampler.py`` does
``from utils.utils import fspecial_gauss`` -- three repo-relative package imports that
don't resolve outside the original repo's directory layout. All three real functions/
classes are inlined verbatim into this single file instead; no code was rewritten.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# utils/utils.py::fspecial_gauss -- unchanged (only reachable via downsample_mode='gauss*',
# not exercised by the default config below, but vendored for fidelity to conv()'s full
# real downsample_mode branch set)
# --------------------------------------------------------------------------------------


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


# --------------------------------------------------------------------------------------
# models/downsampler.py::Downsampler, get_kernel, pad_circular -- unchanged
# --------------------------------------------------------------------------------------


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ["lanczos", "gauss", "box", "uniform", "blur"]

    if phase == 0.5 and kernel_type != "box":
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == "box":
        assert phase == 0.5, "Box filter is always half-phased"
        kernel[:] = 1.0 / (kernel_width * kernel_width)

    elif kernel_type == "gauss":
        assert sigma, "sigma is not specified"
        assert phase != 0.5, "phase 1/2 for gauss not implemented"
        return fspecial_gauss(kernel_width, sigma)

    elif kernel_type == "uniform":
        kernel = np.ones([kernel_width, kernel_width])

    elif kernel_type == "lanczos":
        assert support, "support is not specified"
        center = (kernel_width + 1) / 2.0

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                kernel[i - 1][j - 1] = val
    else:
        assert False, "wrong method name"
    kernel /= kernel.sum()
    return kernel


def pad_circular(x, pad):
    """
    :param x: pytorch tensor of shape: [batch, ch, h, w]
    :param pad: uint
    :return:
    """
    x = torch.cat([x, x[:, :, 0:pad]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad : -pad], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad : -pad], x], dim=3)
    return x


class Downsampler(nn.Module):
    """
    http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(
        self,
        n_planes,
        factor,
        kernel_type,
        phase=0,
        kernel_width=None,
        support=None,
        sigma=None,
        preserve_size=False,
        pad_type="reflection",
        transpose_conv=False,
    ):
        super().__init__()

        assert phase in [0, 0.5], "phase should be 0 or 0.5"

        if kernel_type == "lanczos2":
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "lanczos3":
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "gauss12":
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = "gauss"
        elif kernel_type == "gauss1sq2":
            kernel_width = 9
            sigma = 1.0 / np.sqrt(2)
            kernel_type_ = "gauss"
        elif kernel_type == "uniform_blur":
            kernel_width = 9
            kernel_type_ = "uniform"
            pad_type = "circular"
        elif kernel_type == "gauss_blur":
            kernel_width = 25
            sigma = 1.6
            kernel_type_ = "gauss"
            pad_type = "circular"
        elif kernel_type in {"lanczos", "gauss", "box"}:
            kernel_type_ = kernel_type
        else:
            assert False, "wrong name kernel"

        self.kernel = get_kernel(
            factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma
        )
        if transpose_conv:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) // 2.0)
            else:
                pad = int((self.kernel.shape[0] - factor) // 2.0)
            downsampler = nn.ConvTranspose2d(
                n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=pad
            )
        else:
            downsampler = nn.Conv2d(
                n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0
            )
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:
            if pad_type == "circular":
                self.padding = lambda torch_in: pad_circular(torch_in, kernel_width // 2)
            elif pad_type == "reflection":
                if self.kernel.shape[0] % 2 == 1:
                    pad = int((self.kernel.shape[0] - 1) // 2.0)
                else:
                    pad = int((self.kernel.shape[0] - factor) // 2.0)
                self.padding = nn.ReplicationPad2d(pad)
            else:
                assert False, "pad_type have only circular or reflection options"
        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)


# --------------------------------------------------------------------------------------
# models/common.py -- unchanged
# --------------------------------------------------------------------------------------


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super().__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2 : diff2 + target_shape2, diff3 : diff3 + target_shape3]
                )

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super().__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
    https://arxiv.org/abs/1710.05941
    The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super().__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun="LeakyReLU"):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == "Swish":
            return Swish()
        elif act_fun == "ELU":
            return nn.ELU()
        elif act_fun == "none":
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero", downsample_mode="stride"):
    downsampler = None
    if stride != 1 and downsample_mode != "stride":
        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ["lanczos2", "lanczos3"]:
            downsampler = Downsampler(
                n_planes=out_f,
                factor=stride,
                kernel_type=downsample_mode,
                phase=0.5,
                preserve_size=True,
            )
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------------------
# models/skip.py::skip -- unchanged
# --------------------------------------------------------------------------------------


def skip(
    num_input_channels=2,
    num_output_channels=3,
    num_channels_down=[16, 32, 64, 128, 128],
    num_channels_up=[16, 32, 64, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3,
    filter_size_up=3,
    filter_skip_size=1,
    need_sigmoid=True,
    need_bias=True,
    upsample_mode="nearest",
    use_interpolate=True,
    align_corners=False,
    pad="zero",
    downsample_mode="stride",
    act_fun="LeakyReLU",
    need1x1_up=True,
):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_ = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(
            bn(
                num_channels_skip[i]
                + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
            )
        )

        if num_channels_skip[i] != 0:
            skip_.add(
                conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad)
            )
            skip_.add(bn(num_channels_skip[i]))
            skip_.add(act(act_fun))

        deeper.add(
            conv(
                input_depth,
                num_channels_down[i],
                filter_size_down[i],
                2,
                bias=need_bias,
                pad=pad,
                downsample_mode=downsample_mode[i],
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(
            conv(
                num_channels_down[i],
                num_channels_down[i],
                filter_size_down[i],
                bias=need_bias,
                pad=pad,
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        if use_interpolate:
            deeper.add(
                Interpolate(scale_factor=2, mode=upsample_mode[i], align_corners=align_corners)
            )
        else:
            deeper.add(
                nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=align_corners)
            )

        model_tmp.add(
            conv(
                num_channels_skip[i] + k,
                num_channels_up[i],
                filter_size_up[i],
                1,
                bias=need_bias,
                pad=pad,
            )
        )
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


MENAGERIE_ZOO = "vendored-pytorch"

# Tiny config honoring the repo's real defaults shape (denoising.ipynb::get_network_and_input:
# input_depth=32, skip_n33d=skip_n33u=128, skip_n11=4, num_scales=5, upsample_mode='bilinear',
# downsample_mode='stride', pad='reflection', act_fun='LeakyReLU') -- scaled down for a small
# traced graph: 2 scales, 8/4 channels, 32x32 input.
_INPUT_DEPTH = 2
_OUTPUT_CHANNELS = 3
_HW = 32


def build_deepred_skip():
    return skip(
        num_input_channels=_INPUT_DEPTH,
        num_output_channels=_OUTPUT_CHANNELS,
        num_channels_down=[8, 4],
        num_channels_up=[8, 4],
        num_channels_skip=[2, 2],
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        upsample_mode="bilinear",
        use_interpolate=True,
        align_corners=False,
        pad="reflection",
        downsample_mode="stride",
        act_fun="LeakyReLU",
        need1x1_up=True,
    )


def example_input_deepred_skip():
    # DIP-style fixed random noise input tensor (get_noise(input_depth, 'noise', shape))
    return torch.rand(1, _INPUT_DEPTH, _HW, _HW) * 0.1


MENAGERIE_ENTRIES = [
    (
        "DeepRED (Deep Image Prior skip network)",
        "build_deepred_skip",
        "example_input_deepred_skip",
        2019,
        "vendored-pytorch",
    ),
]
