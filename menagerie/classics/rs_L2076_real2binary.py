# FAITHFUL PORT of https://github.com/larq/zoo @ main (original framework: TensorFlow/Keras + larq)
#
# Real-to-Binary Net (Martinez, Yang, Bulat & Tzimiropoulos, ICLR 2020,
# "Training binary neural networks with real-to-binary convolutions").
#
# The paper's OFFICIAL repo (brais-martinez/real2binary) never published code
# ("Code is still coming...") and instead points readers to a re-implementation
# by the larq team: larq/zoo, file larq_zoo/literature/real_to_bin_nets.py.
# That file is TensorFlow/Keras and depends on `larq` (binary-NN quantized
# layers) and `zookeeper` (config Field/factory system) -- neither is an
# installed base lib here, and installing a second deep-learning framework
# plus its quantization stack is not "vendoring", so this is transcribed
# FAITHFULLY into self-contained base-env torch, translating every mechanism
# 1:1 from the real larq_zoo source:
#
#   - `first_block`: 7x7 stride-2 conv, BN, ReLU (ResNet/R2B use ReLU, not the
#     StrongBaseline's PReLU), 3x3 stride-2 maxpool.
#   - The public `RealToBinaryNet()` factory returns a `RealToBinNetBNNFactory`
#     model: `input_quantizer="ste_sign"`, `kernel_quantizer="ste_sign"` (full
#     BNN variant, both activations and weights binarized).
#   - `block()` = two `half_binary_block`s back to back (8 blocks total across
#     stages 2..9, downsampling when `block % 2 == 0 and block > 3` i.e. at
#     blocks 4, 6, 8), matching the "single real block <-> two binary blocks"
#     supervision scheme from the paper.
#   - `half_binary_block`: shortcut = identity if channel count unchanged,
#     else avgpool(2,stride2)+1x1 conv+BN (`shortcut_connection`); main path =
#     BN -> binary 3x3 conv (STE-sign-binarized activations AND weights,
#     stride 2 iff downsampling) -> data-dependent rescale -> PReLU; output =
#     main path + shortcut.
#   - `RealToBinNetFactory._scale_binary_conv_output` (the "data-dependent"
#     scaling that distinguishes Real-to-Bin nets from the plain
#     StrongBaseline nets, Section 4.3 of the paper): global-avg-pool the
#     *real-valued* conv input -> Linear(reduce, ReLU) -> Linear(expand,
#     Sigmoid) -> reshape -> multiply onto the binary conv output. This is the
#     SE-style "reactivate the real-valued information" step.
#   - `ste_sign` (larq/larq/quantizers.py, `SteSign`): forward = sign(x)
#     (mapping 0 -> +1, matching `tf.math.sign` sign-at-zero convention used
#     by larq's `math.sign` helper for `ste_sign` == 0 -> +1); backward =
#     identity gradient clipped to |x| <= 1 (Straight-Through Estimator).
#   - `last_block`: global average pool -> Linear(num_classes) -> Softmax.
#
# scaling_r (SE reduction ratio) defaults to 8, matching `StrongBaselineNetFactory.scaling_r`.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class _SteSign(torch.autograd.Function):
    """sign(x) forward; straight-through gradient clipped to |x| <= clip_value.

    Faithful port of larq's `ste_sign` (larq/quantizers.py): forward returns
    `sign(x)` (0 maps to +1), backward passes the incoming gradient through
    unchanged wherever |x| <= clip_value and zeroes it elsewhere.
    """

    @staticmethod
    def forward(ctx, x, clip_value=1.0):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        s = torch.sign(x)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= ctx.clip_value).to(grad_output.dtype)
        return grad_output * mask, None


def ste_sign(x, clip_value=1.0):
    return _SteSign.apply(x, clip_value)


class QuantConv2d(nn.Module):
    """Faithful port of `lq.layers.QuantConv2D` as used by RealToBinNetBNNFactory:
    both the input activations and the conv kernel are STE-sign-binarized
    before a standard (real-valued-shape) convolution is applied."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        bin_input = ste_sign(x)
        bin_weight = ste_sign(self.weight)
        return nn.functional.conv2d(bin_input, bin_weight, stride=self.stride, padding=self.padding)


class ShortcutConnection(nn.Module):
    """Faithful port of `_SharedBaseFactory.shortcut_connection`: identity when
    channel counts match, else avgpool(2, stride2) + 1x1 conv (no bias) + BN."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.identity = in_channels == out_channels
        if not self.identity:
            self.pool = nn.AvgPool2d(2, stride=2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)

    def forward(self, x):
        if self.identity:
            return x
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DataDependentRescale(nn.Module):
    """Faithful port of `RealToBinNetFactory._scale_binary_conv_output`: the
    "data-dependent" (SE-style) rescale of the binary conv output using the
    REAL-VALUED conv input (Section 4.3 of Martinez et al. 2020)."""

    def __init__(self, in_channels, out_channels, scaling_r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(1, in_channels // scaling_r)
        self.reduce = nn.Linear(in_channels, reduced, bias=False)
        self.expand = nn.Linear(reduced, out_channels, bias=False)

    def forward(self, conv_input, conv_output):
        z = self.pool(conv_input).flatten(1)
        z = torch.relu(self.reduce(z))
        z = torch.sigmoid(self.expand(z))
        scales = z.view(z.shape[0], z.shape[1], 1, 1)
        return conv_output * scales


class HalfBinaryBlock(nn.Module):
    """Faithful port of `StrongBaselineNetFactory.half_binary_block` with the
    Real-to-Bin `_scale_binary_conv_output` override (data-dependent rescale
    instead of the StrongBaseline's static learned rescale)."""

    def __init__(self, in_channels, downsample, scaling_r=8):
        super().__init__()
        out_channels = in_channels * 2 if downsample else in_channels
        self.shortcut = ShortcutConnection(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = QuantConv2d(
            in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1
        )
        self.rescale = DataDependentRescale(in_channels, out_channels, scaling_r=scaling_r)
        self.prelu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        shortcut_out = self.shortcut(x)
        conv_input = self.bn(x)
        conv_output = self.conv(conv_input)
        x = self.rescale(conv_input, conv_output)
        x = self.prelu(x)
        return x + shortcut_out


class BinaryBlock(nn.Module):
    """Faithful port of `StrongBaselineNetFactory.block`: two half-binary
    blocks back to back, only the first optionally downsampling."""

    def __init__(self, in_channels, downsample, scaling_r=8):
        super().__init__()
        self.block_a = HalfBinaryBlock(in_channels, downsample=downsample, scaling_r=scaling_r)
        out_channels = in_channels * 2 if downsample else in_channels
        self.block_b = HalfBinaryBlock(out_channels, downsample=False, scaling_r=scaling_r)

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        return x


class FirstBlock(nn.Module):
    """Faithful port of `_SharedBaseFactory.first_block` with `use_prelu=False`
    (the Real-to-Bin/ResNet path, as opposed to the StrongBaselineNet path
    which uses PReLU here)."""

    def __init__(self, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class LastBlock(nn.Module):
    """Faithful port of `_SharedBaseFactory.last_block`: global average pool
    -> Linear(num_classes) -> Softmax."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return self.softmax(x)


class RealToBinaryNet(nn.Module):
    """Faithful port of `RealToBinaryNet()` (larq_zoo/literature/real_to_bin_nets.py),
    the public factory function that builds a `RealToBinNetBNNFactory` model
    (the full BNN variant: both activations and weights STE-sign-binarized in
    every block's main conv). 4 `BinaryBlock`s (blocks 2/3, 4/5, 6/7, 8/9 in
    the original numbering), downsampling+doubling channels at the 2nd, 3rd,
    and 4th `BinaryBlock` (original `block % 2 == 0 and block > 3` at blocks
    4, 6, 8) -- i.e. every `BinaryBlock` after the first.
    """

    def __init__(self, num_classes=10, stem_channels=64, scaling_r=8):
        super().__init__()
        self.first_block = FirstBlock(out_channels=stem_channels)
        channels = stem_channels
        stages = []
        for stage_idx in range(4):
            downsample = stage_idx > 0
            stages.append(BinaryBlock(channels, downsample=downsample, scaling_r=scaling_r))
            if downsample:
                channels *= 2
        self.stages = nn.ModuleList(stages)
        self.last_block = LastBlock(channels, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        for stage in self.stages:
            x = stage(x)
        return self.last_block(x)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_NUM_CLASSES = 10
_STEM_CHANNELS = 16  # shrunk from the paper's 64 for a fast tiny-size trace
_INPUT_SIZE = 64


def build_real2binary():
    torch.manual_seed(0)
    model = RealToBinaryNet(num_classes=_NUM_CLASSES, stem_channels=_STEM_CHANNELS)
    model.eval()
    return model


def example_input_real2binary():
    torch.manual_seed(0)
    return torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE)


MENAGERIE_ENTRIES = [
    ("RealToBinaryNet", "build_real2binary", "example_input_real2binary", 2020, MENAGERIE_ZOO),
]
