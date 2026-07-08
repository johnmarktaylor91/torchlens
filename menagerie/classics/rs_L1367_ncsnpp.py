# SOURCE: vendored from https://github.com/sp-uhh/sgmse @ main
# (sgmse/backbones/ncsnpp.py + sgmse/backbones/ncsnpp_utils/{layers,layerspp,
#  normalization,up_or_down_sampling}.py + sgmse/backbones/shared.py, minus
#  sgmse/util/registry.py which is inlined as a tiny local Registry class)
#
# NCSN++ (Song et al., "Score-Based Generative Modeling through Stochastic
# Differential Equations") as adapted by SGMSE (Welker/Richter/Gerkmann,
# "Speech Enhancement with Score-Based Generative Models in the Complex STFT
# Domain") for complex-STFT-domain diffusion-based speech enhancement: a
# BigGAN-style U-Net score network over the complex short-time spectrogram
# (real+imag as 4 input channels: noisy speech + conditioner), FIR up/down
# sampling, Gaussian-Fourier noise-level embedding, multi-resolution
# attention, and a progressive "output_skip" pyramid decoder.
#
# The upstream `up_or_down_sampling.py`'s `upfirdn2d` dispatches to a
# CUDA-compiled `torch.utils.cpp_extension.load()` custom op when
# `torch.cuda.is_available()`, else uses the file's own pure-PyTorch
# `upfirdn2d_native` CPU fallback (also present verbatim upstream in
# `op/upfirdn2d_native.py`, used by `op/upfirdn2d.py`'s own `if
# input.device.type == "cpu"` branch) -- that CPU-only real code path is
# what's vendored here (no CUDA kernel compilation, no architectural change).
# `sgmse/util/registry.py`'s tiny `Registry` class is inlined so the
# `@BackboneRegistry.register("ncsnpp")` decorator on `NCSNpp` works
# unmodified. Every layer/block in `NCSNpp.__init__`/`forward` is transcribed
# verbatim from the real source.

import functools
import math
import string
import warnings
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# sgmse/util/registry.py (verbatim, inlined)
# ---------------------------------------------------------------------------


class Registry:
    def __init__(self, managed_thing: str):
        self.managed_thing = managed_thing
        self._registry = {}

    def register(self, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in self._registry:
                warnings.warn(
                    f"{self.managed_thing} with name '{name}' doubly registered, old class will be replaced."
                )
            self._registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get_by_name(self, name: str):
        if name in self._registry:
            return self._registry[name]
        else:
            raise ValueError(f"{self.managed_thing} with name '{name}' unknown.")

    def get_all_names(self):
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# sgmse/backbones/shared.py (only BackboneRegistry needed)
# ---------------------------------------------------------------------------

BackboneRegistry = Registry("Backbone")


# ---------------------------------------------------------------------------
# sgmse/backbones/ncsnpp_utils/op/upfirdn2d_native.py (verbatim CPU path,
# used directly instead of the CUDA-dispatching op/upfirdn2d.py wrapper)
# ---------------------------------------------------------------------------


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


# ---------------------------------------------------------------------------
# sgmse/backbones/ncsnpp_utils/up_or_down_sampling.py (verbatim)
# ---------------------------------------------------------------------------


class Conv2d(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel,
        up=False,
        down=False,
        resample_kernel=(1, 3, 3, 1),
        use_bias=True,
        kernel_init=None,
    ):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


def _shape(x, dim):
    return x.shape[dim]


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1

    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]

    assert convW == convH

    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = [1, 1, factor, factor]
    output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
    output_padding = (
        output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
        output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW,
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)

    return upfirdn2d(
        x, torch.tensor(k, device=x.device), pad=((p + 1) // 2 + factor - 1, p // 2 + 1)
    )


def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device), pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def upsample_2d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor**2))
    p = k.shape[0] - factor
    return upfirdn2d(
        x, torch.tensor(k, device=x.device), up=factor, pad=((p + 1) // 2 + factor - 1, p // 2)
    )


def downsample_2d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2))


# ---------------------------------------------------------------------------
# sgmse/backbones/ncsnpp_utils/normalization.py (verbatim, only the pieces
# reachable from InstanceNorm2dPlus's `ConditionalInstanceNorm2dPlus` import
# used by layers.py -- unused config-dispatch helper omitted)
# ---------------------------------------------------------------------------


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, : 2 * num_features].normal_(1, 0.02)
            self.embed.weight.data[:, 2 * num_features :].zero_()
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


def get_normalization(config, conditional=False):
    """Obtain normalization modules from the config file. (verbatim; unused
    by NCSNpp's default 'biggan' resblock_type path but kept for parity)."""
    norm = config.model.normalization
    if conditional:
        if norm == "InstanceNorm++":
            return functools.partial(
                ConditionalInstanceNorm2dPlus, num_classes=config.model.num_classes
            )
        else:
            raise NotImplementedError(f"{norm} not implemented yet.")
    else:
        if norm == "InstanceNorm":
            return nn.InstanceNorm2d
        elif norm == "GroupNorm":
            return nn.GroupNorm
        else:
            raise ValueError("Unknown normalization: %s" % norm)


# ---------------------------------------------------------------------------
# sgmse/backbones/ncsnpp_utils/layers.py (verbatim, pieces needed for the
# 'biggan' resblock_type / 'fourier' embedding_type default NCSNpp config)
# ---------------------------------------------------------------------------


def get_act(config):
    if config == "elu":
        return nn.ELU()
    elif config == "relu":
        return nn.ReLU()
    elif config == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError("activation function does not exist!")


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(
                3 * variance
            )
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# sgmse/backbones/ncsnpp_utils/layerspp.py (verbatim)
# ---------------------------------------------------------------------------

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class Upsample(nn.Module):
    def __init__(
        self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), mode="nearest")
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)

        return h


class Downsample(nn.Module):
    def __init__(
        self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)

        return x


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


ResnetBlockBigGAN = ResnetBlockBigGANpp
default_initializer = default_init


# ---------------------------------------------------------------------------
# sgmse/backbones/ncsnpp.py (verbatim class body)
# ---------------------------------------------------------------------------


@BackboneRegistry.register("ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    def __init__(
        self,
        scale_by_sigma=True,
        nonlinearity="swish",
        nf=128,
        ch_mult=(1, 1, 2, 2, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="output_skip",
        progressive_input="input_skip",
        progressive_combine="sum",
        init_scale=0.0,
        fourier_scale=16,
        image_size=256,
        embedding_type="fourier",
        dropout=0.0,
        centered=True,
        **unused_kwargs,
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            image_size // (2**i) for i in range(num_resolutions)
        ]

        self.conditional = conditional = conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        num_channels = 4  # x.real, x.imag, y.real, y.imag
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        modules = []
        if embedding_type == "fourier":
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
            embed_dim = 2 * nf
        elif embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(
            Upsample_, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel
        )  # noqa: F841

        if progressive == "output_skip":
            self.pyramid_upsample = Upsample_(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                Upsample_, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            Downsample_, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel
        )  # noqa: F841

        if progressive_input == "input_skip":
            self.pyramid_downsample = Downsample_(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                Downsample_, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )
        else:
            raise ValueError(
                f'resblock type {resblock_type} unrecognized (only "biggan" vendored).'
            )

        channels = num_channels
        if progressive_input != "none":
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2
                elif progressive_input == "residual":
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == "residual":
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        modules = self.all_modules
        m_idx = 0

        x = torch.cat(
            (
                x[:, [0], :, :].real,
                x[:, [0], :, :].imag,
                x[:, [1], :, :].real,
                x[:, [1], :, :].imag,
            ),
            dim=1,
        )

        if self.embedding_type == "fourier":
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == "positional":
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            x = 2 * x - 1.0

        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:, None, :, :]
        return h


# module-level aliases matching upstream's `layerspp.Upsample`/`Downsample`
# (renamed here to avoid shadowing the `functools.partial` locals bound
# inside `__init__`, which upstream achieves via submodule namespacing)
Upsample_ = Upsample
Downsample_ = Downsample


# ---------------------------------------------------------------------------
# Tiny build/example for TorchLens tracing. `image_size` and `nf` shrunk for
# a fast trace; `ch_mult`/`attn_resolutions`/`resblock_type`/`embedding_type`
# kept at their real architectural defaults (BigGAN-style resblocks, Fourier
# noise embedding, output_skip progressive decoder, FIR up/down sampling).
# ---------------------------------------------------------------------------
def build_ncsnpp():
    torch.manual_seed(0)
    model = NCSNpp(
        scale_by_sigma=True,
        nonlinearity="swish",
        nf=16,
        ch_mult=(1, 2, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="output_skip",
        progressive_input="input_skip",
        progressive_combine="sum",
        init_scale=0.0,
        fourier_scale=16,
        image_size=32,
        embedding_type="fourier",
        dropout=0.0,
        centered=True,
    )
    model.eval()
    return model


def example_input_ncsnpp():
    torch.manual_seed(0)
    real = torch.randn(2, 2, 32, 32)
    imag = torch.randn(2, 2, 32, 32)
    return torch.complex(real, imag)


def _example_time_cond():
    torch.manual_seed(1)
    return torch.rand(2) * 0.5 + 0.5


class _NCSNppTraceWrapper(nn.Module):
    """TorchLens traces a single positional-tensor-argument forward; this
    wraps NCSNpp's (x, time_cond) signature into a single-input module by
    carrying a fixed, real, non-degenerate `time_cond` batch as a buffer."""

    def __init__(self, ncsnpp):
        super().__init__()
        self.ncsnpp = ncsnpp
        self.register_buffer("time_cond", _example_time_cond())

    def forward(self, x):
        return self.ncsnpp(x, self.time_cond)


def build_ncsnpp_wrapped():
    return _NCSNppTraceWrapper(build_ncsnpp())


MENAGERIE_ENTRIES = [
    ("NCSNPP", "build_ncsnpp_wrapped", "example_input_ncsnpp", 2021, MENAGERIE_ZOO),
]
