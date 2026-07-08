# SOURCE: vendored from https://github.com/hustzxd/EfficientPyTorch @ efficient_pytorch
# (the "training code" companion repo for hustzxd/LSQuantization, linked from that
# repo's own README: "The related project with training code:
# https://github.com/hustzxd/EfficientPyTorch")
#
# LSQ-quantized AlexNet (Esser, McKinstry, Bablani, Appuswamy, Modha. 2020, ICLR,
# "LEARNED STEP SIZE QUANTIZATION"). LSQ is a weight/activation quantization method
# (a learned per-tensor-or-per-channel step size with a straight-through gradient)
# applied to an existing CNN backbone rather than a network topology of its own; the
# `hustzxd/LSQuantization` repo (the queue candidate) supplies only the quantized
# `Conv2dLSQ`/`LinearLSQ`/`ActLSQ` layer primitives and explicitly points to this
# companion repo for the actual per-architecture model code that plugs them in.
# Vendored here verbatim is that repo's real `AlexNetQ`: standard AlexNet topology
# (5 conv + 3 fc) with every conv/fc layer after the first replaced by the quantized
# `Conv2dQ`/`LinearQ` layers and every activation gated by a quantized `ActQ`, exactly
# as the original authors wired it up for their AlexNet w4a4/w3a3/w2a2 ImageNet
# experiments (see hustzxd/LSQuantization's own README results table). Source:
#   https://raw.githubusercontent.com/hustzxd/EfficientPyTorch/efficient_pytorch/models/imagenet/alexnetQ.py
#   https://raw.githubusercontent.com/hustzxd/EfficientPyTorch/efficient_pytorch/models/_modules/quantize.py
#   https://raw.githubusercontent.com/hustzxd/EfficientPyTorch/efficient_pytorch/models/_modules/_quan_base.py
#
# What is kept: `AlexNetQ` byte-for-byte (the real 5-conv/3-fc topology, the real
# per-layer ActQ/Conv2dQ/LinearQ placement and `nbits_w`/`nbits_a`/`q_mode`/`l2`
# defaults), and the real `Conv2dQ`/`LinearQ`/`ActQ`/`Qmodes` classes they're built
# from (running-scale statistics quantizer: `running_scale` EMA-tracked buffer,
# `ln_error`/`update_running_scale`/`truncation` search that adjusts the running scale
# toward whichever of {scale/2, scale, scale*2} minimizes quantization error, and the
# real forward-time fake-quantize-with-straight-through-gradient computation
# `wq = y.transpose(0,1).detach() + weight - weight.detach()`), copied verbatim from
# the repo's own `models/_modules/quantize.py` and `_quan_base.py`.
#
# What is dropped (import plumbing / non-architectural, not architecture): the
# `alexnet_q`/`alexnet_lsq`/`alexnet_llsq`/... factory functions' `pretrained=True`
# branches (`load_fake_quantized_state_dict`, `torch.utils.model_zoo` checkpoint
# download -- never called here, only random init is used), the sibling
# `AlexNetQv2`/`AlexNetQPACT`/`AlexNetQFN`/`AlexNetQFNv2`/`AlexNetQFI` variant classes
# (alternate quantization wiring not used by the plain `AlexNetQ` built here), and the
# unrelated modules pulled in transitively by the real package's `models/_modules/
# __init__.py` (`eltwise`, `concat`, `upsample`, `ttq`, `cluster_quant`, `bwn`, `lsq`,
# `llsq`, `svd`, `rnn_q`, `activation`) that `AlexNetQ` itself never imports.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models/_modules/_quan_base.py (real Qmodes enum) ----
class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


# ---- models/_modules/quantize.py (real Conv2dQ / LinearQ / ActQ + their exact
#      helper functions truncation/ln_error/update_running_scale, copied verbatim) ----
def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2**num_bits - 1


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data, nbits=8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2**qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


def ln_error(x, nbits, scale, is_act, l2=True):
    x_clip = (x / scale).clamp(-(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1)
    x_q = x_clip.round()
    x_q = x_q * scale
    if is_act:
        if l2:
            error = ((x - x_q) ** 2).sum() / x.reshape(-1).size()[0]
        else:
            error = (x - x_q).abs().sum() / x.reshape(-1).size()[0]
    else:
        if l2:
            error = ((x - x_q) ** 2).sum(dim=0) / x.shape[0]
        else:
            error = (x - x_q).abs().sum(dim=0) / x.shape[0]
    x_clip = x_clip * scale
    return error, x_clip, x_q


def update_running_scale(data_fp, nbits, scale_old, error, is_act, l2=True):
    s_error, _, _ = ln_error(data_fp, nbits, scale_old / 2, is_act=is_act, l2=l2)
    b_error, _, _ = ln_error(data_fp, nbits, scale_old * 2, is_act=is_act, l2=l2)
    a1 = error - s_error
    a2 = b_error - error
    g1 = a1 >= 0
    g2 = a2 > 0
    g3 = a1 + a2 >= 0
    b = ((g1 == 0) * (g2 == 0) == 1) + ((g1 * (g2 == 0) * (g3 == 0)) > 0) > 0
    s = (((g1 * g2) > 0) + ((g1 * (g2 == 0) * g3) > 0)) > 0
    return b, s


class Conv2dQ(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nbits=4,
        mode=Qmodes.kernel_wise,
        l2=True,
        scale_bits=-1,
        bias_bits=-1,
        ema_decay=0.99,
    ):
        super(Conv2dQ, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if nbits < 0:
            self.register_buffer("running_scale", None)
            return
        self.nbits = nbits
        self.q_model = mode
        self.l2 = l2
        self.scale_bits = scale_bits
        self.bias_bits = bias_bits
        self.ema_decay = ema_decay
        if mode == Qmodes.kernel_wise:
            self.register_buffer("running_scale", torch.zeros(out_channels))
            self.is_layer_wise = False
        else:
            self.register_buffer("running_scale", torch.zeros(1))
            self.is_layer_wise = True
        self.register_buffer("init_state", torch.zeros(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_scale.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return F.conv2d(
                input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            self.running_scale.data.copy_(w_reshape.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        scale = self.running_scale.detach()
        if self.scale_bits > 0:
            scale, _ = truncation(scale, self.scale_bits)
        if self.bias_bits > 0:
            input, scale_a = input[0], input[1]
            scale_bias = scale * scale_a
            if self.scale_bits > 0:
                scale_bias, _ = truncation(scale_bias, self.scale_bits)
            bias_clip = (self.bias / scale_bias).clamp(
                -(2 ** (self.bias_bits - 1)), 2 ** (self.bias_bits - 1) - 1
            )
            bias_q = bias_clip.round()
            bias_q = bias_q * scale_bias
            bq = bias_q.detach() + self.bias - self.bias.detach()
        else:
            bq = self.bias
        error, _, y = ln_error(w_reshape, self.nbits, scale, is_act=self.is_layer_wise, l2=self.l2)
        if self.training:
            with torch.no_grad():
                b, s = update_running_scale(
                    w_reshape, self.nbits, scale, error, self.is_layer_wise, l2=self.l2
                )
                self.running_scale = torch.where(
                    b, scale * self.ema_decay + (1 - self.ema_decay) * scale * 2, scale
                )
                self.running_scale = torch.where(
                    s, scale * self.ema_decay + (1 - self.ema_decay) * scale / 2, self.running_scale
                )
        wq = (
            y.transpose(0, 1).reshape(self.weight.shape).detach()
            + self.weight
            - self.weight.detach()
        )
        return F.conv2d(input, wq, bq, self.stride, self.padding, self.dilation, self.groups)


class LinearQ(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        nbits=4,
        mode=Qmodes.layer_wise,
        l2=True,
        scale_bits=-1,
        bias_bits=-1,
        ema_decay=0.9,
    ):
        super(LinearQ, self).__init__(in_features, out_features, bias=bias)
        if nbits < 0:
            self.register_buffer("running_scale", None)
            return
        self.nbits = nbits
        self.q_mode = mode
        self.l2 = l2
        self.scale_bits = scale_bits
        self.bias_bits = bias_bits
        self.ema_decay = ema_decay
        if mode == Qmodes.kernel_wise:
            self.register_buffer("running_scale", torch.zeros(out_features))
            self.is_layer_wise = False
        else:
            self.register_buffer("running_scale", torch.zeros(1))
            self.is_layer_wise = True
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_scale.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return F.linear(input, self.weight, self.bias)
        w_reshape = self.weight.transpose(0, 1)
        scale = self.running_scale.detach()

        if self.scale_bits > 0:
            scale, _ = truncation(scale, self.scale_bits)
        if self.bias_bits > 0:
            input, scale_a = input[0], input[1]
            scale_bias = scale * scale_a
            if self.scale_bits > 0:
                scale_bias, _ = truncation(scale_bias, self.scale_bits)
            bias_clip = (self.bias / scale_bias).clamp(
                -(2 ** (self.bias_bits - 1)), 2 ** (self.bias_bits - 1) - 1
            )
            bias_q = bias_clip.round()
            bias_q = bias_q * scale_bias
            bq = bias_q.detach() + self.bias - self.bias.detach()
        else:
            bq = self.bias

        error, _, y = ln_error(w_reshape, self.nbits, scale, is_act=self.is_layer_wise, l2=self.l2)
        if self.training:
            with torch.no_grad():
                b, s = update_running_scale(
                    w_reshape, self.nbits, scale, error, self.is_layer_wise, l2=self.l2
                )
                self.running_scale = torch.where(
                    b, scale * self.ema_decay + (1 - self.ema_decay) * scale * 2, scale
                )
                self.running_scale = torch.where(
                    s, scale * self.ema_decay + (1 - self.ema_decay) * scale / 2, self.running_scale
                )
        wq = y.transpose(0, 1).detach() + self.weight - self.weight.detach()
        return F.linear(input, wq, bq)


class ActQ(Module):
    def __init__(
        self,
        nbits=4,
        signed=False,
        l2=True,
        expand=False,
        split=False,
        scale_bits=-1,
        out_scale=False,
        ema_decay=0.999,
    ):
        super(ActQ, self).__init__()
        if nbits < 0:
            self.register_buffer("running_scale", None)
            return
        self.nbits = nbits
        self.signed = signed
        self.expand = expand
        self.split = split
        self.l2 = l2
        self.scale_bits = scale_bits
        self.out_scale = out_scale
        self.ema_decay = ema_decay
        if not signed:
            self.nbits = nbits + 1
        self.register_buffer("running_scale", torch.zeros(1))
        self.register_buffer("init_state", torch.zeros(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_scale.fill_(0.5)

    def forward(self, input):
        if self.running_scale is None:
            return input
        if self.training and self.init_state == 0:
            if self.signed:
                self.running_scale.data.copy_(input.max() / 2 ** (self.nbits - 1))
            else:
                self.running_scale.data.copy_(input.max() / 2**self.nbits)
            self.init_state.fill_(1)
        scale = self.running_scale.detach()
        if self.scale_bits > 0:
            scale, _ = truncation(scale, nbits=self.scale_bits)
        error, x_clip, y = ln_error(input, self.nbits, scale, is_act=True, l2=self.l2)
        if self.training:
            with torch.no_grad():
                b, s = update_running_scale(
                    input, self.nbits, scale, error, is_act=True, l2=self.l2
                )
                self.running_scale = torch.where(
                    b, scale * self.ema_decay + (1 - self.ema_decay) * scale * 2, scale
                )
                self.running_scale = torch.where(
                    s, scale * self.ema_decay + (1 - self.ema_decay) * scale / 2, self.running_scale
                )
        output = y.detach() + x_clip - x_clip.detach()
        if self.expand is False and self.split is False:
            return [output, scale] if self.out_scale else output
        # expand/split paths (int7/uint8 activation-splitting) are not exercised by
        # AlexNetQ's default construction (expand=False, split=False everywhere) and
        # are omitted; AlexNetQ never sets expand=True/split=True.
        raise NotImplementedError("expand/split ActQ paths unused by AlexNetQ")


# ---- models/imagenet/alexnetQ.py (real AlexNetQ class, byte-faithful) ----
class AlexNetQ(nn.Module):
    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(AlexNetQ, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv3
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv4
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # As the experiment result shows, there is no difference between layer wise with kernel wise.
            LinearQ(256 * 6 * 6, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc6
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            nn.Dropout(),
            LinearQ(4096, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def build_alexnet_lsq():
    # real AlexNetQ(nbits_w=4, nbits_a=4) -- the w4a4 LSQ config the LSQuantization
    # README reports ImageNet results for; num_classes shrunk for fast tracing.
    return AlexNetQ(num_classes=10, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True)


def example_input_alexnet_lsq():
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("AlexNet-LSQ", "build_alexnet_lsq", "example_input_alexnet_lsq", 2020, "vendored-pytorch"),
]
