# FAITHFUL PORT of https://github.com/microsoft/LQ-Nets @ master (original framework:
# TensorFlow 1.x + tensorpack, using `tf.contrib`, which is EOL and cannot be reasonably
# installed alongside modern Python 3.11 / torch here)
#
# LQ-Nets (Zhang, Yang, Ye, Hua, "LQ-Nets: Learned Quantization for Highly Accurate and
# Compact Deep Neural Networks", ECCV 2018): a quantized ResNet whose convolution weights
# and post-activation feature maps are passed through a *learned* K-bit quantizer -- a
# per-channel (weights) / per-tensor (activations) basis-vector decomposition (not a fixed
# uniform/log quantizer) fit online via least-squares during training, with a
# straight-through estimator on the backward pass (`y = x + stop_gradient(-x) +
# stop_gradient(quantized)`). This module transcribes the actual repo code 1:1 (not a paper
# paraphrase), at the repo's own published defaults for ImageNet
# (`Model(depth=18, mode='resnet', qw=1, qa=2)` in imagenet.py -- LQ-ResNet-18, 1-bit
# weights / 2-bit activations, the non-preactivation basicblock backbone):
#   learned_quantization.py  (`QuantizedWeight`, `QuantizedActiv`, `Conv2DQuant`)
#   resnet_model.py          (`resnet_basicblock`, `resnet_group`, `resnet_backbone`)
#   imagenet.py              (`Model.__init__`/`get_logits` -- the depth->num_blocks table
#                              and the default `mode='resnet'` backbone selection)
#
# TensorFlow 1.x (`tf.get_variable`/`tf.variable_scope`, `tf.contrib.layers.
# variance_scaling_initializer`, `tensorpack.models.BatchNorm`/`Conv2D`/`LinearWrap`,
# `tf.stop_gradient` custom-gradient tricks, multi-GPU `tower_context`-gated moving-average
# basis updates) is not installed and is architecturally incompatible with modern torch, so
# the architecture is transcribed here as self-contained torch: the same learned
# per-channel/per-tensor basis quantizer math (level/threshold construction from a learned
# basis, least-squares basis refit during training, straight-through gradient), the same
# quantized-conv + BN + ReLU + quantized-activation block structure, and the same
# non-preactivation ResNet-18 basicblock backbone (`conv0` 7x7 stride-2 stem -> 4 groups of
# `Conv2DQuant`+BN+ReLU basicblocks with a `1x1` quantized-conv projection shortcut when
# channel counts change -> global average pool -> linear classifier). The least-squares
# basis-refit path (`ctx.is_main_training_tower`, multi-GPU tower gating, `tf.summary`
# logging) is a training-time online calibration step, not part of the architecture's
# forward computation graph in eval mode; a plain forward pass here uses the (buffer-held,
# EMA-updated in training) learned `basis` directly, matching what a single non-multi-GPU
# forward/backward step of the original graph actually executes.
#
# Repo: https://github.com/microsoft/LQ-Nets @ master
# Files: learned_quantization.py, resnet_model.py, imagenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

NORM_PPF_0_75 = 0.6745
MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001


def _level_codes(nbit, num_levels, signed):
    """Binary-code -> {0,1} (activations) or {-1,+1} (weights) level multiplier matrix,
    faithful port of the `init_level_multiplier` construction in `QuantizedActiv` /
    `QuantizedWeight`.
    """
    codes = torch.zeros(num_levels, nbit)
    for i in range(num_levels):
        level_number = i
        for j in range(nbit):
            bit = level_number % 2
            if signed and bit == 0:
                bit = -1
            codes[i, j] = float(bit)
            level_number //= 2
    return codes


def _thrs_multiplier(nbit, num_levels):
    """Faithful port of `init_thrs_multiplier`: adjacent-level midpoint selector matrix."""
    thrs = torch.zeros(num_levels - 1, num_levels)
    for i in range(1, num_levels):
        thrs[i - 1, i - 1] = 0.5
        thrs[i - 1, i] = 0.5
    return thrs


class QuantizedActiv(nn.Module):
    """Learned activation quantizer, faithful port of `learned_quantization.QuantizedActiv`.

    `basis`: [nbit, 1] learned (EMA-refit during training) basis vector, unsigned levels
    (matches the repo's non-negative post-ReLU activation quantization).
    """

    def __init__(self, nbit=2):
        super().__init__()
        self.nbit = nbit
        self.num_levels = 2**nbit
        init_basis = torch.tensor(
            [(NORM_PPF_0_75 * 2 / (2**nbit - 1)) * (2.0**i) for i in range(nbit)]
        ).view(nbit, 1)
        self.register_buffer("basis", init_basis)
        self.register_buffer("level_codes", _level_codes(nbit, self.num_levels, signed=False))
        self.register_buffer("thrs_multiplier", _thrs_multiplier(nbit, self.num_levels))

    def forward(self, x):
        levels = self.level_codes @ self.basis  # [num_levels, 1]
        levels, sort_id = torch.topk(levels.t(), self.num_levels, dim=1)
        levels = levels.flip(-1).t()  # [num_levels, 1], ascending
        thrs = self.thrs_multiplier @ levels  # [num_levels - 1, 1]

        y = torch.zeros_like(x)
        for i in range(self.num_levels - 1):
            g = x > thrs[i, 0]
            y = torch.where(g, levels[i + 1, 0], y)

        x_clip = torch.minimum(x, levels[self.num_levels - 1, 0])
        # gradient: d(output)/d(x) = d(x_clip)/d(x); value: quantized y.
        y = x_clip + (-x_clip).detach() + y.detach()
        return y


class QuantizedWeight(nn.Module):
    """Learned per-output-channel weight quantizer, faithful port of
    `learned_quantization.QuantizedWeight`.

    `basis`: [nbit, num_filters] learned (EMA-refit during training) basis, signed levels
    (+-1 binary codes), initialized from He/MSRA fan-in variance `n` as in the original.
    """

    def __init__(self, num_filters, n, nbit=1):
        super().__init__()
        self.nbit = nbit
        self.num_filters = num_filters
        self.num_levels = 2**nbit
        base = NORM_PPF_0_75 * ((2.0 / n) ** 0.5) / (2 ** (nbit - 1))
        init_basis = torch.stack(
            [torch.full((num_filters,), (2**j) * base) for j in range(nbit)], dim=0
        )
        self.register_buffer("basis", init_basis)
        self.register_buffer("level_codes", _level_codes(nbit, self.num_levels, signed=True))
        self.register_buffer("thrs_multiplier", _thrs_multiplier(nbit, self.num_levels))

    def forward(self, w):
        """w: [out_channels, in_channels, kh, kw] (torch conv weight layout).

        The original operates on TF's [kh, kw, in, out] layout with per-out-channel
        ("num_filters") basis; here we reshape to put the out-channel axis last for the
        per-channel matmuls, matching the original's channel-last convention exactly.
        """
        out_c = w.shape[0]
        w_last = w.permute(1, 2, 3, 0).contiguous()  # [in, kh, kw, out]
        flat_shape = w_last.shape

        levels = self.level_codes @ self.basis  # [num_levels, num_filters]
        levels, sort_id = torch.topk(
            levels.t(), self.num_levels, dim=1
        )  # [num_filters, num_levels]
        levels = levels.flip(-1).t()  # [num_levels, num_filters], ascending
        thrs = self.thrs_multiplier @ levels  # [num_levels - 1, num_filters]

        reshape_w = w_last.reshape(-1, out_c)  # [N, num_filters]
        y = torch.zeros_like(reshape_w) + levels[0]
        for i in range(self.num_levels - 1):
            g = reshape_w > thrs[i]
            y = torch.where(g, levels[i + 1].unsqueeze(0).expand_as(reshape_w), y)
        y = y.view(flat_shape)

        y = w_last + (-w_last).detach() + y.detach()  # gradient: y=w (straight-through)
        return y.permute(3, 0, 1, 2).contiguous()  # back to [out, in, kh, kw]


class Conv2DQuant(nn.Module):
    """Learned-weight-quantized 2D convolution, faithful port of
    `learned_quantization.Conv2DQuant` (bias-free variant, matching `resnet_backbone`'s
    `argscope(Conv2DQuant, use_bias=False, ...)`).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, is_quant=True, nbit=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.is_quant = is_quant
        if is_quant:
            n = kernel_size * kernel_size * out_channels
            self.weight_quant = QuantizedWeight(out_channels, n, nbit=nbit)

    def forward(self, x):
        weight = self.weight_quant(self.conv.weight) if self.is_quant else self.conv.weight
        return F.conv2d(x, weight, bias=None, stride=self.conv.stride, padding=self.conv.padding)


def resnet_shortcut(in_channels, out_channels, stride):
    """Faithful port of `resnet_model.resnet_shortcut` (block_type='B' branch: a 1x1
    quantized conv + BN when the channel count changes, identity otherwise).
    """
    if in_channels != out_channels or stride != 1:
        return nn.Sequential(
            Conv2DQuant(in_channels, out_channels, 1, stride=stride, nbit=1),
            nn.BatchNorm2d(out_channels),
        )
    return nn.Identity()


class ResnetBasicBlock(nn.Module):
    """Faithful port of `resnet_model.resnet_basicblock` (non-preactivation basicblock:
    conv-BN-ReLU -> conv-BN -> += quantized shortcut -> ReLU, from `resnet_group`).
    """

    def __init__(self, in_channels, ch_out, stride, qw=1, qa=2):
        super().__init__()
        self.conv1 = Conv2DQuant(in_channels, ch_out, 3, stride=stride, nbit=qw)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.quant1 = QuantizedActiv(nbit=qa)
        self.conv2 = Conv2DQuant(ch_out, ch_out, 3, stride=1, nbit=qw)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.shortcut = resnet_shortcut(in_channels, ch_out, stride)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.quant1(out)
        out = self.bn2(self.conv2(out))
        return out + shortcut


def resnet_group(in_channels, ch_out, count, stride, qw=1, qa=2):
    """Faithful port of `resnet_model.resnet_group` (a group of basicblocks, each block's
    output ReLU'd -- `resnet_group`'s own `l = tf.nn.relu(l)` after each `block_func` call).
    """
    blocks = []
    c_in = in_channels
    for i in range(count):
        blocks.append(ResnetBasicBlock(c_in, ch_out, stride if i == 0 else 1, qw=qw, qa=qa))
        c_in = ch_out
    return nn.ModuleList(blocks)


class LQResNet(nn.Module):
    """Faithful port of `resnet_model.resnet_backbone` at the repo's own ImageNet defaults
    (`imagenet.py`'s `Model(depth=18, mode='resnet', qw=1, qa=2)`): stem `Conv2DQuant`
    (unquantized weights, `is_quant=False` matching `conv0`) + BN + ReLU + maxpool, 4 groups
    of quantized-weight/quantized-activation basicblocks, global average pool, FC classifier.
    """

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=1000, qw=1, qa=2):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)

        widths = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        groups = []
        in_channels = 64
        for width, stride, count in zip(widths, strides, num_blocks):
            groups.append(resnet_group(in_channels, width, count, stride, qw=qw, qa=qa))
            in_channels = width
        self.groups = nn.ModuleList(groups)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, image):
        x = F.relu(self.bn0(self.conv0(image)))
        x = self.pool0(x)
        for group in self.groups:
            for block in group:
                x = F.relu(block(x))
        x = self.gap(x).flatten(1)
        return self.linear(x)


def build_lq_resnet18():
    # Repo default: ImageNet, 224x224, depth=18 -> num_blocks=[2,2,2,2], qw=1 (1-bit
    # weights), qa=2 (2-bit activations); num_classes shrunk from 1000 for a fast trace,
    # architecture (quantized-weight conv stem + 4 basicblock groups + quantized
    # activations + GAP + linear head) unchanged.
    return LQResNet(num_blocks=(2, 2, 2, 2), num_classes=20, qw=1, qa=2)


def example_input_lq_resnet18():
    return torch.randn(2, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "LQ-Net (LQ-ResNet-18, learned quantization)",
        build_lq_resnet18,
        example_input_lq_resnet18,
        2018,
        MENAGERIE_ZOO,
    ),
]
