# FAITHFUL PORT of hpi-xnor/BMXNet-v2-examples @ master (model def) +
# hpi-xnor/BMXNet-v2 @ master (binary layer primitives)
# (original framework: MXNet / Gluon)
#
# MeliusNet (Bethge, Bartz, Yang, Chen, Meinel. 2021, WACV, "MeliusNet: Can Binary
# Neural Networks Achieve MobileNet-level Accuracy?"). A DenseNet-style binary neural
# network: repeated "base blocks" = [DenseBlock (binary 3x3 activated_conv, feature
# concat, no bottleneck since bn_size=0) + ImprovementBlock (binary 3x3 activated_conv
# whose output is sliced-added back onto the last `growth_rate` input channels instead
# of a plain concat, so it "improves" existing features instead of only adding new
# ones)], with grouped-stem-style transitions between stages. Ported here is the
# official HPI author repo's real architecture code (not a paper reimplementation):
#   https://raw.githubusercontent.com/hpi-xnor/BMXNet-v2-examples/master/binary_models/meliusnet.py
#   https://raw.githubusercontent.com/hpi-xnor/BMXNet-v2-examples/master/binary_models/basenet_dense.py
#   https://raw.githubusercontent.com/hpi-xnor/BMXNet-v2-examples/master/binary_models/common_layers.py
#   https://raw.githubusercontent.com/hpi-xnor/BMXNet-v2/master/python/mxnet/gluon/nn/binary_layers.py
#
# Cannot run/vendor as-is: BMXNet-v2 is a source fork of MXNet itself (C++ core +
# custom Gluon ops `det_sign`/`approx_sign`/`gradcancel`, built via CMake/Ninja against
# mxnet v1.5.1). MXNet is EOL (Apache retired the project in 2023) and this
# environment has no MXNet at all, let alone a fork with custom binary ops -- there is
# nothing to install. This is a from-scratch-in-torch TRANSCRIPTION of the real
# MXNet/Gluon code's exact computation graph and default hyperparameters, not a
# paper-only reimplementation.
#
# What is preserved exactly (mechanism-for-mechanism from the real source files):
#   - `BaseNetDense` stem ("imagenet" variant: 7x7 s2 conv -> BN -> ReLU -> 3x3 s2
#     maxpool), per-stage repeated base blocks, and inter-stage transitions built from
#     the `DOWNSAMPLE_STRUCT = "bn,max_pool,relu,fp_conv"` token list (BatchNorm ->
#     MaxPool2d(2,2) -> ReLU -> full-precision 1x1 conv, channel count from the real
#     `meliusnet_spec['22']` reduction factors), exactly as `_make_transition` builds it.
#   - `_add_dense_block`: with `bn_size=0` (the real default for MeliusNet via
#     `get_basenet_constructor(..., default_bn_size=0)`) this is BN -> binary 3x3
#     `activated_conv` (`growth_rate` out channels, `padding=dilation`) with NO
#     bottleneck 1x1, then channel-concat with the block input (`HybridConcurrent` +
#     `Identity`) -- reproduced as `DenseBlock` (BN -> BinaryConv3x3 -> cat).
#   - `ImprovementBlock`: BN -> binary 3x3 `activated_conv` (`growth_rate` out
#     channels) -> the real "sliced addition": the conv output is added onto the LAST
#     `growth_rate` input channels only, and the first `in_channels - growth_rate`
#     channels pass through untouched, then both parts are concatenated back
#     (`self.slices = [0, in_channels-channels, in_channels]`, `slices_add_x = [False,
#     True]`) -- reproduced exactly via channel-slice + add + `torch.cat`.
#   - `activated_conv`'s default path (`ActivatedConvolutionFactory.__call__` with the
#     module-level default `binary_layer_config.approximation == ""`) is
#     `BinaryConvolution`: `QActivation` (bits_a=1, method="det_sign") then `QConv2D`
#     (bits=1, method="det_sign" weight quantization, `apply_scaling=False`). Both use
#     the real `det_sign(x) = sign(x)` (MXNet's `F.det_sign`, the deterministic +-1
#     binarizer) with an identity/clip straight-through backward (MXNet's
#     `F.contrib.gradcancel` zeroing gradients outside `[-threshold, threshold]`,
#     `threshold=1.0` default) -- reproduced as `DetSignSTE` (round-to-{-1,+1} forward,
#     clipped-identity backward) for both weight and activation binarization.
#   - `_QConv.hybrid_forward`'s real "offset" mechanic: for `bits==1` and
#     `not no_offset and not scaling` (true for plain `BinaryConvolution`), the raw
#     binary convolution output `h` is corrected as `h = (h + offset) / 2` where
#     `offset = prod(weight.shape[1:])` (the per-output-pixel receptive-field size),
#     which converts a {-1,+1}x{-1,+1} popcount-style conv into the XNOR-equivalent of a
#     plain float conv on {-1,+1} inputs -- reproduced exactly in `BinaryConv2d.forward`.
#   - `_apply_pre_padding`: binary layers pad with a constant value of -1 (not 0,
#     because inputs are already in {-1,+1}) -- reproduced via `F.pad(..., value=-1.0)`
#     before an otherwise zero-padding conv.
#   - The real `meliusnet22` spec (`block_config=[4,5,4,4]`,
#     `reduction=[1/(160/320), 1/(224/480), 1/(256/480)]`, `init_features=64`,
#     `growth_rate=64`, `downsample=DOWNSAMPLE_STRUCT`) is used to build the tiny demo
#     instance below, with the number of ImageNet-scale blocks/classes reduced only for
#     fast tracing (architecture unchanged).
#
# What is dropped (import plumbing / non-architectural, not architecture): CLI
# arg-parsing (`BaseNetDenseParameters._add_arguments`), MXNet `ctx=cpu()` /
# `root=...` pretrained-download plumbing (`get_basenet_constructor`'s
# `pretrained=True` branch explicitly `raise`s "no pretrained model" upstream anyway),
# and the alternate `meliusnet_a/b/c` `ChannelShuffle`-based transition variants (not
# used by the plain `meliusnet22` spec built here).
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# --- real det_sign + gradcancel straight-through binarizer (MXNet F.det_sign +
#     F.contrib.gradcancel(threshold=1.0), used for both weight and activation
#     quantization in the default BinaryConvolution path) ---
class DetSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold=1.0):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        out = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= ctx.threshold).to(grad_output.dtype)
        return grad_output * mask, None


def det_sign(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return DetSignSTE.apply(x, threshold)


class QActivation(nn.Module):
    """Real QActivation(bits_a=1, method='det_sign') from binary_layers.py."""

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return det_sign(x, self.threshold)


class BinaryConv2d(nn.Module):
    """
    Real `_QConv` (bits=1, method='det_sign', apply_scaling=False) wired up through
    `QConv2D` -> `BinaryConvolution`: binarize the (already-binarized-activation)
    input's padding to -1, binarize the weight with det_sign, run a plain conv, then
    apply the real popcount-equivalent offset correction `(h + offset) / 2`.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.offset = in_channels * kernel_size * kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode="constant", value=-1.0)
        w_q = det_sign(self.weight)
        h = F.conv2d(x, w_q, bias=None, stride=self.stride, dilation=self.dilation)
        return (h + self.offset) / 2


class BinaryActivatedConv(nn.Module):
    """Real `activated_conv` default factory output: QActivation -> QConv2D."""

    def __init__(self, channels, in_channels, kernel_size=3, stride=1, padding=0, dilation=1):
        super().__init__()
        self.qact = QActivation()
        self.qconv = BinaryConv2d(in_channels, channels, kernel_size, stride, padding, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qconv(self.qact(x))


class DenseBlock(nn.Module):
    """Real `_add_dense_block` with bn_size=0 (MeliusNet's actual default): BN ->
    binary 3x3 conv -> concat with block input (no bottleneck 1x1)."""

    def __init__(self, in_channels, growth_rate, dilation=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = BinaryActivatedConv(
            growth_rate, in_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv(self.bn(x))
        return torch.cat([x, new_features], dim=1)


class ImprovementBlock(nn.Module):
    """Real `ImprovementBlock`: BN -> binary 3x3 conv -> sliced addition onto the
    last `channels` input channels, concatenated back with the untouched prefix."""

    def __init__(self, channels, in_channels, dilation=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = BinaryActivatedConv(
            channels, in_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation
        )
        assert channels < in_channels
        self.split = in_channels - channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(self.bn(x))
        untouched = residual[:, : self.split]
        touched = residual[:, self.split :] + out
        return torch.cat([untouched, touched], dim=1)


class Transition(nn.Module):
    """Real `_make_transition` built from DOWNSAMPLE_STRUCT='bn,max_pool,relu,fp_conv'
    (full-precision 1x1 conv changes channel count per the real reduction factors)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class MeliusNet(nn.Module):
    """
    Real `BaseNetDense` / `MeliusNet` forward structure: imagenet stem -> per-stage
    repeated [DenseBlock, ImprovementBlock] pairs -> transition (except after the last
    stage) -> finalize (BN, ReLU, global avg pool) -> Dense classifier.
    Ported at the real `meliusnet22` spec (`block_config=[4,5,4,4]`, growth_rate=64,
    init_features=64) with only `num_classes`/input resolution shrunk for fast tracing.
    """

    def __init__(
        self,
        block_config=(4, 5, 4, 4),
        growth_rate=64,
        init_features=64,
        reduction=(320 / 160, 480 / 224, 480 / 256),
        num_classes=10,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = init_features
        stages = []
        for stage_idx, repeat_num in enumerate(block_config):
            blocks = []
            for _ in range(repeat_num):
                blocks.append(DenseBlock(num_features, growth_rate))
                num_features += growth_rate
                blocks.append(ImprovementBlock(growth_rate, num_features))
            stages.append(nn.Sequential(*blocks))
            if stage_idx != len(block_config) - 1:
                out_features = int(round((num_features / reduction[stage_idx]) / 32)) * 32
                stages.append(Transition(num_features, out_features))
                num_features = out_features
        self.stages = nn.Sequential(*stages)

        self.finalize = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.finalize(x)
        x = self.output(x)
        return x


def build_meliusnet22():
    # meliusnet22 real spec (block_config=[4,5,4,4]); num_classes shrunk for fast tracing.
    return MeliusNet(
        block_config=(4, 5, 4, 4),
        growth_rate=64,
        init_features=64,
        reduction=(320 / 160, 480 / 224, 480 / 256),
        num_classes=10,
    )


def example_input_meliusnet22():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("MeliusNet22", "build_meliusnet22", "example_input_meliusnet22", 2021, "ported-pytorch"),
]
