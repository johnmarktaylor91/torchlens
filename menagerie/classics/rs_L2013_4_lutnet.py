# FAITHFUL PORT of awai54st/LUTNet @ master (tiled-lutnet/training-software)
# (original framework: Keras 2 (old `keras.layers.normalization` API) + TensorFlow 1.x)
#
# LUTNet (Wang, Lu, Fraser, Blott, Constantinides. 2019, FCCM, "LUTNet: Rethinking
# Inference in FPGA Soft Logic" / tiled follow-up "LUTNet: Learning FPGA
# Configurations for Highly Efficient Neural Network Inference"). A binary neural
# network whose convolution/dense layers are NOT plain XNOR-popcount binary layers but
# a differentiable emulation of an FPGA K-LUT: each output activation is produced by a
# *learned truth table* (a 5-input Lagrangian interpolation over 32 trainable
# {-1,+1}-ish coefficients `c1..c32`, gated by a learned "BRAM content" bit `w1` and
# three fixed random input-shuffle maps) rather than a single learned weight -- this
# is the architecture's entire point (the FPGA LUT primitive has more expressive power
# per resource than a single XNOR gate). The queue candidate `hpi-xnor/LUTNet` URL is
# a 404; the real, maintained repo is `awai54st/LUTNet` (same LUTNet paper authors,
# Xilinx/Imperial). Ported here is the CIFAR-10 config's real, complete architecture
# code from the "tiled-lutnet" (2nd paper) variant of the official repo, not a
# from-scratch reimplementation from the paper text:
#   https://raw.githubusercontent.com/awai54st/LUTNet/master/tiled-lutnet/training-software/model_architectures.py
#   https://raw.githubusercontent.com/awai54st/LUTNet/master/tiled-lutnet/training-software/binarization_utils.py
#
# Cannot run/vendor as-is: the real code is Keras 2 (old `keras.layers.normalization`
# and `keras.engine.topology.Layer` APIs, both removed from modern Keras/tf.keras) on
# TensorFlow 1.x graph-mode ops (`tf.extract_image_patches`, `K.get_session()`,
# `tf.gather_nd` used as a manual "advanced indexing" workaround the code comments
# call out as a TF1 limitation). Neither TF1 nor old-API Keras 2 is installed in this
# (torch-only) environment, and pinning them would be a dead, unmaintained stack. This
# is a from-scratch-in-torch TRANSCRIPTION of the real Keras/TF code's exact
# computation graph and default hyperparameters (`resid_levels=2`, `LUT=True`,
# `BINARY=True`, per the repo's own `lutnet_training_script.sh` invocation
# `python Binary.py CIFAR-10 True False True False True True True 50`, arg order
# `dataset resid_levels(=True->2 via script) ... LUT=True ...`), not a paper-only
# reimplementation.
#
# What is preserved exactly (mechanism-for-mechanism from the real source files):
#   - `Residual_sign` (`levels=2`): two-level residual sign decomposition -- level 0 is
#     `binarize(x) * |mean0|`, level 1 is `binarize(x - level0) * |mean1|`, stacked
#     into an extra leading "levels" axis exactly as `tf.stack([out_bin, out])` does.
#     `binarize(x) = clip(x,-1,1)` forward-rounded to `sign()` with an identity
#     straight-through gradient (the repo's own "trick from Sergey Ioffe" comment),
#     reproduced as `BinarizeSTE`.
#   - `binary_conv`/`binary_dense` with `levels==1 or first_layer==True` (the real
#     first-conv-layer special case: "1st layer cannot use LUTNet architecture
#     because inputs are always in fxp"): a plain binarized-weight conv/linear, no LUT
#     machinery, exactly as the `if self.levels==1 or self.first_layer==True` branch.
#   - `binary_conv`/`binary_dense` with `levels==2, LUT=True, BINARY=True` (the real
#     architecture's actual layers, per the CIFAR-10 `get_model` call sites which all
#     pass `LUT=LUT, BINARY=BINARY` with the script's `LUT=True, BINARY=True`): the
#     real 5-input Lagrangian-interpolation LUT emulation --
#       * three FIXED random "shuffle" index maps (`rand_map_exp_0/1/2`, one random
#         permutation per map, sampled once at construction and held fixed
#         `trainable=False`, exactly as `np.random.randint(window_size, size=...)`)
#         that select three "neighbor" input elements per output-window position;
#       * the current element and its 3 shuffled neighbors are each split into a
#         `pos`/`neg` pair via `(1 +/- binarize(v))/2` (the real "Lagrangian
#         interpolating polynomial" comment in the source) -- for the *outer* x-select
#         axis this pair is additionally scaled by `abs(x)` (the trainable multi-level
#         residual magnitude), for the 3 shuffled neighbors it is NOT (matching the
#         source's `#*abs(...)` commented-out scaling on the shuffled terms exactly);
#       * a learned "BRAM content" weight `w1` is likewise split into `ws0_pos/neg`
#         via the same `(1 +/- binarize(w1))/2` construction;
#       * the output is the sum over all 2^5=32 sign combinations of (x-select-pol,
#         shuffle0-pol, shuffle1-pol, shuffle2-pol, w-pol) of
#         `product-of-the-4-input-pol-terms * corresponding-c-mask * w-pol-term`,
#         where the 32 `c1..c32` learned masks map 1:1 onto the 32 sign combinations
#         in the real code's exact binary-counting order (x-select is the MSB, w-pol
#         is the LSB) -- reproduced here by *generating* that exact 32-term sum via
#         `itertools.product` over the 5 boolean axes rather than manually retyping
#         the source's 32 unrolled `self.out = self.out + ...` lines, which are
#         mechanically equivalent to this loop (verified against the real source's
#         literal term ordering: c1='pos,pos,pos,pos,pos', c2='pos,pos,pos,pos,neg',
#         ..., c32='neg,neg,neg,neg,neg' with w-pol alternating pos/neg every term);
#       * this whole term is then repeated for BOTH residual levels (`x[0]` and
#         `x[1]`, i.e. `levels=2`) and summed, exactly as the real code's `x0_*`/`x1_*`
#         (conv) or `x[0,:,:]`/`x[1,:,:]` (dense) blocks.
#     For the conv layer the window im2col is done via `nn.Unfold` (torch's
#     `F.unfold` is the exact equivalent of TF's `tf.extract_image_patches` used
#     here); the fixed shuffle maps are applied on the unfolded per-window feature
#     axis, matching the real `tf.gather_nd` indexing.
#   - `binary_conv`'s real per-layer trainable `gamma` scale (`constraint_gamma =
#     |gamma|`, multiplying every `c` mask), initialized to 1.0 exactly as `K.variable(1.0)`.
#   - The real CIFAR-10 stack from `model_architectures.py`'s `get_model` CIFAR-10
#     branch: `binary_conv(64,3->64,k3,valid,first_layer) -> BN -> ResidSign ->
#     binary_conv(64->64,LUT) -> BN -> MaxPool(2,2) -> ResidSign -> binary_conv(64->128,LUT)
#     -> BN -> ResidSign -> binary_conv(128->128,LUT) -> BN -> MaxPool(2,2) -> ResidSign
#     -> binary_conv(128->256,LUT) -> BN -> ResidSign -> binary_conv(256->256,LUT) -> BN
#     -> flatten -> ResidSign -> binary_dense(->512,LUT) -> BN -> ResidSign ->
#     binary_dense(512->512,LUT) -> BN -> ResidSign -> binary_dense(512->10,LUT) -> BN
#     -> softmax`, with the real `valid`-padding 3x3 convs (no explicit padding, exactly
#     as `padding='valid'` in every `binary_conv(...)` call in the source).
#
# What is dropped (import plumbing / non-architectural, not architecture): the
# `pruning_mask`/`pruning_prob` machinery (`add_weight(..., trainable=False)`
# initialized to all-ones in the real code's own default `bnn_pruning.py`-populated
# checkpoints before any pruning has actually been applied -- the mask multiplies
# every weight/LUT-mask by 1.0 with `pruning_prob` only affecting a separate offline
# pruning script not invoked here, so omitting it changes no computation for a
# freshly-constructed model); the CLI training driver (`Binary.py`), the `bnn_pruning.py`
# / `lutnet_init.py` checkpoint-surgery scripts, and matplotlib/pickle plotting
# helpers in `binarization_utils.py` that `get_model` never touches.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# --- real `binarize(x) = clip(x,-1,1)` forward-rounded with identity straight-through
#     gradient ("Sergey Ioffe trick" per the source's own comment) ---
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        clipped = x.clamp(-1, 1)
        return torch.sign(clipped)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize(x: torch.Tensor) -> torch.Tensor:
    return BinarizeSTE.apply(x)


class ResidualSign(nn.Module):
    """Real `Residual_sign(levels=2)`: two-level residual sign decomposition with
    trainable per-level magnitude means, stacked into a leading axis."""

    def __init__(self, num_channels: int, levels: int = 2):
        super().__init__()
        assert levels == 2
        ars = torch.arange(levels).float() + 1.0
        ars = ars.flip(0)
        means = ars / ars.sum()
        self.means = nn.Parameter(means.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        out0 = binarize(resid) * self.means[0].abs()
        resid = resid - out0
        out1 = binarize(resid) * self.means[1].abs()
        return torch.stack([out0, out1], dim=0)  # (levels=2, N, C, ...)


def _lut32_terms(
    x_terms: list[torch.Tensor],
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    c_masks: list[torch.Tensor],
) -> torch.Tensor:
    """
    Real 32-term Lagrangian-interpolation LUT sum, generated (not hand-unrolled) to
    exactly match the source's literal term ordering: x-select is bit 0 (MSB across
    the 4 x-position/shuffle axes), w-select is the final bit (LSB), and the c-mask
    index counts up in that same binary order (c1..c32). Each term is a real
    `K.dot(x_masked, c_masked)` matrix product over the window/in-feature axis
    (`...w -> ...o` via the shared `w` window axis), summed across all 32 terms --
    exactly as the source's `self.out = self.out + K.dot(...)` accumulation.

    `x_terms` = [ (x_pos, x_neg), (xs0_pos, xs0_neg), (xs1_pos, xs1_neg), (xs2_pos, xs2_neg) ],
    each a (pos, neg) pair of tensors of shape (..., window).
    `w_pos`/`w_neg`: (window, out). `c_masks`: 32 x (window, out).
    """
    total = None
    idx = 0
    for bits in itertools.product([0, 1], repeat=4):
        x_prod = None
        for (pos, neg), bit in zip(x_terms, bits):
            term = pos if bit == 0 else neg
            x_prod = term if x_prod is None else x_prod * term
        for w_bit in (0, 1):
            w_term = w_pos if w_bit == 0 else w_neg
            c = c_masks[idx]
            contribution = torch.matmul(x_prod, c * w_term)
            total = contribution if total is None else total + contribution
            idx += 1
    return total


class LUTConv2d(nn.Module):
    """
    Real `binary_conv` (levels=2, LUT=True, BINARY=True): a 3x3/valid-padding LUT
    convolution built from im2col (`F.unfold`, the torch equivalent of the source's
    `tf.extract_image_patches`) + the 32-term Lagrangian LUT sum, summed over the two
    residual levels, exactly as the real `binary_conv.call`'s `levels==2, LUT==True`
    branch (`x0_*`/`x1_*` blocks).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        window_size = in_channels * kernel_size * kernel_size

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.randn(window_size, out_channels) * (1.0 / window_size**0.5))
        self.c = nn.Parameter(torch.randn(32, window_size, out_channels) * (1.0 / window_size**0.5))
        # fixed random shuffle maps (trainable=False in the real code)
        self.register_buffer("rand_map_0", torch.randint(0, window_size, (window_size,)))
        self.register_buffer("rand_map_1", torch.randint(0, window_size, (window_size,)))
        self.register_buffer("rand_map_2", torch.randint(0, window_size, (window_size,)))

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        # x2: (levels=2, N, C_in, H, W)
        levels, n, c_in, h, w = x2.shape
        constraint_gamma = self.gamma.abs()
        clamped_c = constraint_gamma * binarize(self.c)  # (32, window, out)
        clamped_w1 = binarize(self.w1)  # (window, out)
        ws0_pos = (1 + clamped_w1) / 2
        ws0_neg = (1 - clamped_w1) / 2

        out_h = h - self.k + 1
        out_w = w - self.k + 1

        outs = []
        for level in range(levels):
            patches = F.unfold(x2[level], kernel_size=self.k)  # (N, window, L)
            patches = patches.transpose(1, 2)  # (N, L, window)

            shuf0 = patches[:, :, self.rand_map_0]
            shuf1 = patches[:, :, self.rand_map_1]
            shuf2 = patches[:, :, self.rand_map_2]

            x_pos = (1 + binarize(patches)) / 2 * patches.abs()
            x_neg = (1 - binarize(patches)) / 2 * patches.abs()
            xs0_pos = (1 + binarize(shuf0)) / 2
            xs0_neg = (1 - binarize(shuf0)) / 2
            xs1_pos = (1 + binarize(shuf1)) / 2
            xs1_neg = (1 - binarize(shuf1)) / 2
            xs2_pos = (1 + binarize(shuf2)) / 2
            xs2_neg = (1 - binarize(shuf2)) / 2

            x_terms = [(x_pos, x_neg), (xs0_pos, xs0_neg), (xs1_pos, xs1_neg), (xs2_pos, xs2_neg)]
            c_masks = list(clamped_c)  # 32 x (window, out)
            level_out = _lut32_terms(x_terms, ws0_pos, ws0_neg, c_masks)  # (N, L, out)
            outs.append(level_out)

        total = outs[0] + outs[1]
        total = total.transpose(1, 2)  # (N, out, L)
        return total.reshape(n, self.out_channels, out_h, out_w)


class LUTLinear(nn.Module):
    """
    Real `binary_dense` (levels=2, LUT=True, BINARY=True): the same 32-term LUT
    Lagrangian sum, applied to a flattened feature vector via matmul instead of
    im2col, exactly as `binary_dense.call`'s `levels==2, LUT==True` branch.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.randn(in_features, out_features) * (1.0 / in_features**0.5))
        self.c = nn.Parameter(torch.randn(32, in_features, out_features) * (1.0 / in_features**0.5))
        self.register_buffer("rand_map_0", torch.randint(0, in_features, (in_features,)))
        self.register_buffer("rand_map_1", torch.randint(0, in_features, (in_features,)))
        self.register_buffer("rand_map_2", torch.randint(0, in_features, (in_features,)))

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        # x2: (levels=2, N, in_features)
        levels, n, f = x2.shape
        constraint_gamma = self.gamma.abs()
        clamped_c = constraint_gamma * binarize(self.c)
        clamped_w1 = binarize(self.w1)
        ws0_pos = (1 + clamped_w1) / 2
        ws0_neg = (1 - clamped_w1) / 2

        outs = []
        for level in range(levels):
            v = x2[level]  # (N, in_features)
            shuf0 = v[:, self.rand_map_0]
            shuf1 = v[:, self.rand_map_1]
            shuf2 = v[:, self.rand_map_2]

            x_pos = (1 + binarize(v)) / 2 * v.abs()
            x_neg = (1 - binarize(v)) / 2 * v.abs()
            xs0_pos = (1 + binarize(shuf0)) / 2
            xs0_neg = (1 - binarize(shuf0)) / 2
            xs1_pos = (1 + binarize(shuf1)) / 2
            xs1_neg = (1 - binarize(shuf1)) / 2
            xs2_pos = (1 + binarize(shuf2)) / 2
            xs2_neg = (1 - binarize(shuf2)) / 2

            x_terms = [(x_pos, x_neg), (xs0_pos, xs0_neg), (xs1_pos, xs1_neg), (xs2_pos, xs2_neg)]
            c_masks = list(clamped_c)
            level_out = _lut32_terms(x_terms, ws0_pos, ws0_neg, c_masks)  # (N, out)
            outs.append(level_out)

        return outs[0] + outs[1]


class FirstLayerBinaryConv2d(nn.Module):
    """Real `binary_conv(..., first_layer=True)`: plain binarized-weight conv, no LUT
    machinery ("1st layer cannot use LUTNet architecture because inputs are always in
    fxp")."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            * (1.0 / (in_channels * kernel_size * kernel_size) ** 0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clamped_w = self.gamma.abs() * binarize(self.weight)
        return F.conv2d(x, clamped_w)


class LUTNetCIFAR10(nn.Module):
    """
    Real CIFAR-10 `get_model` stack (tiled-lutnet, `resid_levels=2, LUT=True,
    BINARY=True`): first_layer conv -> BN -> [ResidSign -> LUT-conv -> BN]* with
    2x2/stride-2 maxpools after every 2 conv stages -> flatten -> ResidSign ->
    [LUT-dense -> BN -> ResidSign]* -> LUT-dense(->num_classes) -> BN -> softmax.
    Channel widths/kernel counts and valid-padding convs are the real spec; input
    resolution/num_classes are the standard CIFAR-10 32x32x3 / 10-class real config
    (kept as-is; nothing shrunk for tracing since CIFAR-10 is already small).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = FirstLayerBinaryConv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.rs1 = ResidualSign(64)
        self.conv2 = LUTConv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.rs2 = ResidualSign(64)
        self.conv3 = LUTConv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.rs3 = ResidualSign(128)
        self.conv4 = LUTConv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.rs4 = ResidualSign(128)
        self.conv5 = LUTConv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.rs5 = ResidualSign(256)
        self.conv6 = LUTConv2d(256, 256, 3)
        self.bn6 = nn.BatchNorm2d(256)

        # flattened feature size for a 32x32 CIFAR-10 input through 6 valid 3x3 convs
        # and 2 stride-2 maxpools: 32 -> conv1(30) -> conv2(28) -> pool(14) ->
        # conv3(12) -> conv4(10) -> pool(5) -> conv5(3) -> conv6(1); 256*1*1
        self._flat_features = 256 * 1 * 1

        self.rs6 = ResidualSign(1)
        self.fc1 = LUTLinear(self._flat_features, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.rs7 = ResidualSign(1)
        self.fc2 = LUTLinear(512, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.rs8 = ResidualSign(1)
        self.fc3 = LUTLinear(512, num_classes)
        self.bn9 = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x2 = self.rs1(x)
        x = self.conv2(x2)
        x = self.bn2(x)
        x = self.pool1(x)

        x2 = self.rs2(x)
        x = self.conv3(x2)
        x = self.bn3(x)
        x2 = self.rs3(x)
        x = self.conv4(x2)
        x = self.bn4(x)
        x = self.pool2(x)

        x2 = self.rs4(x)
        x = self.conv5(x2)
        x = self.bn5(x)
        x2 = self.rs5(x)
        x = self.conv6(x2)
        x = self.bn6(x)

        x = x.reshape(x.size(0), -1)

        x2 = self.rs6(x)
        x = self.fc1(x2)
        x = self.bn7(x)
        x2 = self.rs7(x)
        x = self.fc2(x2)
        x = self.bn8(x)
        x2 = self.rs8(x)
        x = self.fc3(x2)
        x = self.bn9(x)
        return F.softmax(x, dim=-1)


def build_lutnet_cifar10():
    return LUTNetCIFAR10(num_classes=10)


def example_input_lutnet_cifar10():
    return torch.randn(2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "LUTNet-CIFAR10",
        "build_lutnet_cifar10",
        "example_input_lutnet_cifar10",
        2019,
        "ported-pytorch",
    ),
]
