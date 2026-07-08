# FAITHFUL PORT of Hyungjun-K1m/BinaryDuo @ d23ad764298ee354a7afe7a2508922c256bcd9d9
# (original framework: Torch7 / Lua)
#
# BinaryDuo (Kim, Park, Yoo. ICLR 2020, "BinaryDuo: Reducing Gradient Mismatch in
# Binary Activation Network by Coupling Binary Activations"). BinaryDuo trains a
# 2-bit ("coupled") binary-activation network, then splits ("decouples") each coupled
# neuron into two 1-bit binary neurons for inference, halving the gradient mismatch of
# naive 1-bit binarized-activation networks while keeping pure {0,1} activations at
# deploy time. Ported here is the repo's own `tinyVGG7_decoupled` model (the tiny
# VGG-7 CIFAR-10 network, the repo's smallest and best fully self-contained example of
# the decoupling mechanism), transcribed faithfully from the real Torch7 model file:
#   https://raw.githubusercontent.com/Hyungjun-K1m/BinaryDuo/master/models/tinyVGG7_decoupled.lua
# and its real activation-quantization layer:
#   https://raw.githubusercontent.com/Hyungjun-K1m/BinaryDuo/master/newLayers/QuantizedNeurons.lua
#
# Cannot run/vendor as-is: Torch7/Lua (cutorch/cudnn/nn Lua packages) is not installable
# in this (or any modern) Python/PyTorch environment -- Torch7 is long abandoned
# upstream. This is a from-scratch-in-torch TRANSCRIPTION of the real Lua code's exact
# computation graph, not a paper-only reimplementation.
#
# What is preserved exactly (mechanism-for-mechanism from the real .lua files):
#   - The "decoupling" trick: after each conv+BN, `nn.Replicate(2,1,3)` +
#     `nn.Reshape(2*C, H, W)` duplicates the channel dimension (dim=1) BEFORE the
#     shared BatchNorm+activation are applied, giving each duplicated copy identical
#     input but letting the trained BN affine params later be split into two distinct
#     per-half biases at decouple time -- reproduced here as `x.repeat(1, 2, 1, 1)`
#     immediately followed by the real `nn.SpatialBatchNormalization` (2*C channels)
#     and the real 1-bit activation.
#   - The real 1-bit `QuantizedNeurons` activation: `add_activation(model, opt.activation,
#     1)` in the Lua file hardcodes `Abits=1` for every stage of tinyVGG7_decoupled, so
#     `self.n = 2^1 - 1 = 1`; with `mode=0` (not the -1/+1 mode) and `exception=false`
#     (the default in the Lua ctor), `updateOutput` reduces exactly to
#     `output = round(clamp(input, 0, 1) * 1) / 1`, i.e. round-to-{0,1} on the
#     `HardTanh(0,1)`-clamped input, with an identity straight-through backward
#     (`gradInput:copy(gradOutput)`) -- reproduced here as `HardTanhQuantizeSTE`.
#   - The real conv/pool/BN topology and channel/kernel/stride/padding sizes at every
#     stage (3->45->90x2, 90->45->90x2, 90->90->180x2, 180->90->180x2, then the two
#     1x1-equivalent "FC-as-conv" stages 180->336->672 and 672->512->10), including the
#     real final `Convolution(180,336,4,4,4,4,0,0)` global-pool-as-conv and the final
#     `nn.View(10)` (Torch7's flatten, here `x.flatten(1)`), and the trailing plain
#     `ReLU` (not quantized) before the last 1x1 classifier conv.
#   - Real weight init (`ConvInit`: `normal(0, sqrt(4/n))` with
#     `n = kW*kH*nInputPlane + ceil(kW/dW)*ceil(kH/dH)*nOutputPlane`, bias=None/zeroed;
#     `BNInit`: weight=1, bias=0) is reproduced for faithfulness even though TorchLens
#     traces at random init regardless.
#
# What is dropped (harness/data plumbing, not architecture): CUDA-specific cudnn
# module aliasing, the `opt.tensorType`/`opt.cudnn` deterministic-mode toggles, and the
# `dataloader.lua`/`train.lua`/`checkpoints.lua` training harness -- none of that is
# part of the trainable network graph.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class HardTanhQuantizeSTE(torch.autograd.Function):
    """Faithful port of Torch7 `QuantizedNeurons:updateOutput`/`updateGradInput` with
    the tinyVGG7_decoupled call site's fixed args (bitA=1, binarymode=0, exception=false):
    forward = round(clamp(input, 0, 1) * 1) / 1 == round(clamp(input, 0, 1)); backward
    is the identity straight-through estimator (`gradInput:copy(gradOutput)`)."""

    @staticmethod
    def forward(ctx, x):
        clamped = x.clamp(0.0, 1.0)
        return clamped.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantized_binary_activation(x):
    return HardTanhQuantizeSTE.apply(x)


class DecoupleActivation(nn.Module):
    """`nn.HardTanh(0,1,true); QuantizedNeurons(1,0)` -- the real 1-bit
    ClippedReLU1 + straight-through binary quantizer used at every stage."""

    def forward(self, x):
        return quantized_binary_activation(x)


class DecoupleStage(nn.Module):
    """One decoupled conv stage: real conv -> (optional real 2x2 max pool) ->
    `nn.Replicate(2,1,3)` + `nn.Reshape` (channel-doubling duplication, here
    `repeat(1,2,1,1)`) -> real BatchNorm2d(2*out_channels) -> real 1-bit activation.
    Matches the Lua file's per-stage block exactly."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else None
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.act = DecoupleActivation()

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        x = x.repeat(1, 2, 1, 1)  # nn.Replicate(2,1,3) + nn.Reshape(2*C,H,W,true)
        x = self.bn(x)
        x = self.act(x)
        return x


class TinyVGG7Decoupled(nn.Module):
    """Faithful port of `models/tinyVGG7_decoupled.lua`'s `createModel` (the tiny
    VGG-7 CIFAR-10 network)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # model:add(Convolution(3,45,3,3,1,1,1,1)); Replicate+Reshape(90,32,32); BN(90); act
        self.stage1 = DecoupleStage(3, 45, kernel_size=3, stride=1, padding=1, pool=False)
        # model:add(Convolution(90,45,3,3,1,1,1,1)); Max(2,2,2,2); Replicate+Reshape(90,16,16); BN(90); act
        self.stage2 = DecoupleStage(90, 45, kernel_size=3, stride=1, padding=1, pool=True)
        # model:add(Convolution(90,90,3,3,1,1,1,1)); Max(2,2,2,2); Replicate+Reshape(180,8,8); BN(180); act
        self.stage3 = DecoupleStage(90, 90, kernel_size=3, stride=1, padding=1, pool=True)
        # model:add(Convolution(180,90,3,3,1,1,1,1)); Max(2,2,2,2); Replicate+Reshape(180,4,4); BN(180); act
        self.stage4 = DecoupleStage(180, 90, kernel_size=3, stride=1, padding=1, pool=True)
        # model:add(Convolution(180,336,4,4,4,4,0,0)); Replicate+Reshape(672,1,1); BN(672); act
        self.stage5 = DecoupleStage(180, 336, kernel_size=4, stride=4, padding=0, pool=False)

        # model:add(Convolution(672,512,1,1,1,1,0,0)); BN(512); ReLU(true)
        self.conv6 = nn.Conv2d(672, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        # model:add(Convolution(512,10,1,1,1,1,0,0)); model:add(nn.View(10))
        self.conv7 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Faithful port of ConvInit (normal(0, sqrt(4/n)), n = kW*kH*nInputPlane +
        # ceil(kW/dW)*ceil(kH/dH)*nOutputPlane; bias removed/zeroed) and BNInit
        # (weight=1, bias=0). Random-init tracing does not depend on this, but it is
        # kept for architectural faithfulness.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kw, kh = m.kernel_size
                dw, dh = m.stride
                n = (
                    kw * kh * m.in_channels
                    + math.ceil(kw / dw) * math.ceil(kh / dh) * m.out_channels
                )
                nn.init.normal_(m.weight, 0, math.sqrt(4.0 / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = x.flatten(1)  # nn.View(10)
        return x


def build_binaryduo():
    return TinyVGG7Decoupled(num_classes=10).eval()


def example_input_binaryduo():
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("BinaryDuo", "build_binaryduo", "example_input_binaryduo", 2020, "ported-pytorch"),
]
