# FAITHFUL PORT of enyac-group/single-path-nas @ master (original framework: TensorFlow 1.12 / tf.keras)
# https://raw.githubusercontent.com/enyac-group/single-path-nas/master/nas-search/superkernel.py
# https://raw.githubusercontent.com/enyac-group/single-path-nas/master/nas-search/singlepath_supernet.py
# https://raw.githubusercontent.com/enyac-group/single-path-nas/master/nas-search/supernet_macro.py
# https://raw.githubusercontent.com/enyac-group/single-path-nas/master/nas-search/pixel1_runtime_model.json
#
# Stamoulis, Ding, Wang, Lymberopoulos, Priyantha, Liu, Marculescu, 2019
# (arXiv:1904.02877) "Single-Path NAS: Designing Hardware-Efficient ConvNets
# in less than 4 Hours". The repo's README pins "Tensorflow 1.12, Python
# 3.5+" and the code is tf.keras/tf.layers/tf.variable_scope TF1.x
# throughout (confirmed live: `nas-search/*.py`) -- TF1.x cannot be
# installed alongside this task's torch env, so this is a faithful port
# (real TF1.x source code transcribed mechanism-for-mechanism into torch),
# not a from-scratch reimplementation from the paper.
#
# Single-Path NAS's headline architectural contribution IS the differentiable
# "superkernel" (`superkernel.py::DepthwiseConv2DMasked`): a single 5x5
# depthwise-conv weight tensor is masked at every forward pass by two nested,
# straight-through-estimator threshold indicators -- (1) a 3x3-vs-5x5
# effective-kernel-size mask (learned threshold `t5x5` gates whether the
# outer ring of the 5x5 kernel contributes) and (2) a 50%-vs-100% channel-
# expansion mask (learned thresholds `t50c`/`t100c` gate which channel slice
# of the (already kernel-masked) weight contributes) -- so kernel-size AND
# expansion-ratio search happen inside ONE single-path forward pass with no
# discrete branching or auxiliary supernet paths (the paper's whole point,
# contrasted with DARTS/ProxylessNAS-style multi-path supernets). This is
# ported verbatim below as `DepthwiseConv2DMasked` (renamed `SuperKernelConv2d`
# to avoid the TF-specific "Conv2D subclass" framing, same masking math).
#
# `singlepath_supernet.py::MBConvBlock`/`SinglePathSuperNet` (renamed
# `SinglePathBlock`/`SinglePathSuperNet`) are ported verbatim: the MnasNet-
# style inverted-residual macro architecture (expand -> superkernel depthwise
# -> project, with residual add when shapes match) driving the searchable
# blocks. `supernet_macro.py::single_path_search()`'s exact `blocks_args`
# string list (the concrete MBConv macro-architecture actually used to
# instantiate the search supernet in the real repo, not a guessed config)
# is decoded via a transcribed `MBConvDecoder` and used to build the network
# below. `nas-search/pixel1_runtime_model.json` (the real per-layer Pixel-1
# on-device latency lookup table used only for the auxiliary differentiable
# runtime-regularization scalar the paper adds to the training loss -- it
# does not affect the classification logits) is inlined verbatim as
# `_PIXEL1_RUNTIME_LUT`.
#
# Translation notes (framework differences, not architecture changes):
# - NHWC (TF default) -> NCHW (torch default); conv/BN dims transposed
#   accordingly, no mechanism added or removed.
# - `tf.stop_gradient((x>=0) - sigmoid(x)) + sigmoid(x)` (the paper's
#   straight-through binary-indicator estimator) -> an equivalent
#   `torch.autograd.Function` with a sigmoid-gradient backward, verbatim
#   in effect.
# - `padding='same'` (TF) computed explicitly as static padding for the
#   fixed input resolution used here (stride-1 stem/head 1x1/3x3 convs use
#   the same integer padding TF's 'same' would produce for odd kernels).
# - The `runtime_lut`-driven `total_runtime` scalar is carried through and
#   returned as in the original `call()`, faithfully reproducing the
#   auxiliary differentiable-runtime term (not used for anything except
#   being returned, exactly like the TF version's `total_runtime` return).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

# ============================================================================
# nas-search/pixel1_runtime_model.json (verbatim; real on-device Pixel-1
# latency LUT indexed by [block_idx][candidate_idx in {0,1,2,3}])
# ============================================================================

_PIXEL1_RUNTIME_LUT = {
    "20": {"1": 4.9216999999999995, "0": 2.4381, "3": 5.410299999999999, "2": 2.6649},
    "21": {"1": 6.499499999999999, "0": 6.0662, "3": 7.0644, "2": 6.3308},
    "1": {"1": 9.2015, "0": 4.4905, "3": 11.5892, "2": 5.9248},
    "0": {"1": 3.9452, "0": 3.9844, "3": 3.9277, "2": 3.8584},
    "3": {"1": 6.351099999999999, "0": 3.4223, "3": 9.1756, "2": 4.972099999999999},
    "2": {"1": 6.402200000000001, "0": 3.4111, "3": 9.1518, "2": 4.8853},
    "5": {"1": 4.0999, "0": 1.9592999999999998, "3": 5.116999999999999, "2": 2.4458},
    "4": {"1": 6.4176, "0": 3.4216, "3": 9.156699999999999, "2": 4.987500000000001},
    "7": {"1": 3.9802000000000004, "0": 2.0502999999999996, "3": 5.0630999999999995, "2": 2.6166},
    "6": {
        "1": 3.982999999999999,
        "0": 2.0488999999999997,
        "3": 5.058199999999999,
        "2": 2.6193999999999997,
    },
    "9": {"1": 2.4647, "0": 1.2319999999999998, "3": 2.7908999999999997, "2": 1.3958},
    "8": {"1": 4.0257000000000005, "0": 2.0762, "3": 5.107199999999999, "2": 2.5942},
    "11": {"1": 3.1346, "0": 1.5784999999999998, "3": 3.6589, "2": 1.8073999999999997},
    "10": {"1": 3.1394999999999995, "0": 1.5805999999999996, "3": 3.6309, "2": 1.8193},
    "13": {"1": 3.3782, "0": 1.7338999999999998, "3": 3.918599999999999, "2": 1.9648999999999999},
    "12": {"1": 3.1416, "0": 1.5868999999999995, "3": 3.6603, "2": 1.8032},
    "15": {"1": 4.4261, "0": 2.1805, "3": 5.1933, "2": 2.4549000000000003},
    "14": {"1": 4.3918, "0": 2.1805, "3": 5.0673, "2": 2.4366999999999996},
    "17": {"1": 3.3410999999999995, "0": 1.6163, "3": 3.5496999999999996, "2": 1.6996000000000002},
    "16": {"1": 4.479999999999999, "0": 2.1937999999999995, "3": 5.0946, "2": 2.4534999999999996},
    "19": {"1": 4.879700000000001, "0": 2.4598, "3": 5.3248999999999995, "2": 2.6466999999999996},
    "18": {"1": 4.8923, "0": 2.4304, "3": 5.139399999999999, "2": 2.5696999999999997},
}


# ============================================================================
# superkernel.py :: Indicator (straight-through binary threshold estimator)
# ============================================================================


class _IndicatorFunction(torch.autograd.Function):
    """`tf.stop_gradient((x>=0) - sigmoid(x)) + sigmoid(x)`: forward is the
    hard step function `(x >= 0)`, backward flows through as if it were
    `sigmoid(x)` (straight-through estimator)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * sig * (1 - sig)


def indicator(x):
    return _IndicatorFunction.apply(x)


# ============================================================================
# superkernel.py :: DepthwiseConv2DMasked (ported as SuperKernelConv2d)
# ============================================================================


class SuperKernelConv2d(nn.Module):
    """Single-path differentiable depthwise superkernel: one 5x5 depthwise
    weight, masked at every forward pass into an effective {3x3, 5x5} kernel
    size AND {50%, 100%} channel-expansion slice via learned thresholds.
    For kernel_size != 5 this degrades to a plain depthwise conv (matching
    the real `custom = False` branch)."""

    def __init__(self, channels, kernel_size, stride, runtimes=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.custom = kernel_size == 5

        if not self.custom:
            self.depthwise_kernel = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
            _conv_kernel_init(self.depthwise_kernel)
            self.padding = kernel_size // 2
            return

        self.depthwise_kernel = nn.Parameter(torch.empty(channels, 1, 5, 5))
        _conv_kernel_init(self.depthwise_kernel)
        self.padding = 2

        # thresholds (real repo: add_weight(shape=(1,), initializer='zeros'))
        self.t5x5 = nn.Parameter(torch.zeros(1))
        self.t50c = nn.Parameter(torch.zeros(1))
        self.t100c = nn.Parameter(torch.zeros(1))

        # static masks over the (channels, 1, 5, 5) kernel
        mask3x3 = torch.zeros(channels, 1, 5, 5)
        mask3x3[:, :, 1:4, 1:4] = 1.0
        mask5x5 = 1.0 - mask3x3
        self.register_buffer("mask3x3", mask3x3)
        self.register_buffer("mask5x5", mask5x5)

        c50 = int(round(1.0 * channels / 2.0))
        c100 = int(round(2.0 * channels / 2.0))
        mask50c = torch.zeros(channels, 1, 5, 5)
        mask50c[0:c50] = 1.0
        mask100c = torch.zeros(channels, 1, 5, 5)
        mask100c[c50:c100] = 1.0
        self.register_buffer("mask50c", mask50c)
        self.register_buffer("mask100c", mask100c)

        runtimes = runtimes if runtimes is not None else [0.0, 0.0, 0.0, 0.0]
        self.R50c = float(runtimes[2])
        self.R100c = float(runtimes[3])
        self.R5x5 = float(runtimes[3])
        self.R3x3 = float(runtimes[1])
        self._has_50c_drop = stride == 1 and len(runtimes) == 5

    def forward(self, x, total_runtime):
        if not self.custom:
            out = F.conv2d(
                x,
                self.depthwise_kernel,
                stride=self.stride,
                padding=self.padding,
                groups=self.channels,
            )
            return out, total_runtime

        kernel_3x3 = self.depthwise_kernel * self.mask3x3
        kernel_5x5 = self.depthwise_kernel * self.mask5x5
        norm5x5 = torch.norm(kernel_5x5)

        x5x5 = norm5x5 - self.t5x5
        d5x5 = indicator(x5x5)

        masked_outside = kernel_3x3 + kernel_5x5 * d5x5

        kernel_50c = masked_outside * self.mask50c
        kernel_100c = masked_outside * self.mask100c
        norm50c = torch.norm(kernel_50c)
        norm100c = torch.norm(kernel_100c)

        x100c = norm100c - self.t100c
        d100c = indicator(x100c)

        if self._has_50c_drop:
            x50c = norm50c - self.t50c
            d50c = indicator(x50c)
        else:
            d50c = torch.ones(1, dtype=x.dtype, device=x.device)

        depthwise_kernel_masked = d50c * (kernel_50c + d100c * kernel_100c)

        out = F.conv2d(
            x,
            depthwise_kernel_masked,
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )

        ratio = self.R3x3 / self.R5x5 if self.R5x5 != 0 else 0.0
        runtime_channels = d50c * (self.R50c + d100c * (self.R100c - self.R50c))
        runtime = runtime_channels * ratio + runtime_channels * (1 - ratio) * d5x5
        total_runtime = total_runtime + runtime

        return out, total_runtime


def _conv_kernel_init(weight):
    # conv_kernel_initializer: normal(0, sqrt(2 / fan_out)), fan_out computed
    # from (kh, kw, out_filters) as in the real repo's TF initializer.
    out_channels = weight.shape[0]
    kh, kw = weight.shape[-2], weight.shape[-1]
    fan_out = kh * kw * out_channels
    with torch.no_grad():
        weight.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))


def _dense_kernel_init(weight):
    # dense_kernel_initializer: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
    fan_in = weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        weight.uniform_(-bound, bound)


# ============================================================================
# singlepath_supernet.py :: BlockArgs / GlobalParams / round_filters
# ============================================================================


class BlockArgs:
    __slots__ = (
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    )

    def __init__(
        self,
        kernel_size,
        num_repeat,
        input_filters,
        output_filters,
        expand_ratio,
        id_skip,
        strides,
        se_ratio=None,
    ):
        self.kernel_size = kernel_size
        self.num_repeat = num_repeat
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip
        self.strides = strides
        self.se_ratio = se_ratio

    def _replace(self, **kwargs):
        d = {k: getattr(self, k) for k in self.__slots__}
        d.update(kwargs)
        return BlockArgs(**d)


def round_filters(filters, depth_multiplier=None, depth_divisor=8, min_depth=None):
    if not depth_multiplier:
        return filters
    filters *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


# ============================================================================
# supernet_macro.py :: MBConvDecoder._decode_block_string + single_path_search
# ============================================================================


def _decode_block_string(block_string):
    import re

    ops = block_string.split("_")
    options = {}
    for op in ops:
        splits = re.split(r"(\d.*)", op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value
    return BlockArgs(
        kernel_size=int(options["k"]),
        num_repeat=int(options["r"]),
        input_filters=int(options["i"]),
        output_filters=int(options["o"]),
        expand_ratio=int(options["e"]),
        id_skip=("noskip" not in block_string),
        se_ratio=float(options["se"]) if "se" in options else None,
        strides=[int(options["s"][0]), int(options["s"][1])],
    )


def single_path_search():
    """Verbatim `supernet_macro.py::single_path_search()` blocks_args list
    (the real search-supernet macro-architecture used in the repo)."""
    blocks_args_strings = [
        "r1_k3_s11_e1_i32_o16_noskip",
        "r4_k5_s22_e6_i16_o24",
        "r4_k5_s22_e6_i24_o40",
        "r4_k5_s22_e6_i40_o80",
        "r4_k5_s11_e6_i80_o96",
        "r4_k5_s22_e6_i96_o192",
        "r1_k3_s11_e6_i192_o320_noskip",
    ]
    return [_decode_block_string(s) for s in blocks_args_strings]


# ============================================================================
# singlepath_supernet.py :: MBConvBlock (ported as SinglePathBlock)
# ============================================================================


class SinglePathBlock(nn.Module):
    def __init__(self, block_args, layer_runtimes, bn_momentum=0.99, bn_eps=1e-3):
        super().__init__()
        self.block_args = block_args
        self.has_expand = block_args.expand_ratio != 1
        expanded = block_args.input_filters * block_args.expand_ratio

        if self.has_expand:
            self.expand_conv = nn.Conv2d(
                block_args.input_filters, expanded, kernel_size=1, stride=1, padding=0, bias=False
            )
            _conv_kernel_init(self.expand_conv.weight)
            self.bn0 = nn.BatchNorm2d(expanded, momentum=1 - bn_momentum, eps=bn_eps)

        self.depthwise_conv = SuperKernelConv2d(
            expanded,
            block_args.kernel_size,
            tuple(block_args.strides)[0]
            if isinstance(block_args.strides, (list, tuple))
            else block_args.strides,
            runtimes=layer_runtimes,
        )
        self.stride = block_args.strides[0]
        self.bn1 = nn.BatchNorm2d(expanded, momentum=1 - bn_momentum, eps=bn_eps)

        self.project_conv = nn.Conv2d(
            expanded, block_args.output_filters, kernel_size=1, stride=1, padding=0, bias=False
        )
        _conv_kernel_init(self.project_conv.weight)
        self.bn2 = nn.BatchNorm2d(block_args.output_filters, momentum=1 - bn_momentum, eps=bn_eps)

    def forward(self, x, total_runtime):
        inputs = x
        if self.has_expand:
            x = F.relu(self.bn0(self.expand_conv(x)))

        x, total_runtime = self.depthwise_conv(x, total_runtime)
        x = F.relu(self.bn1(x))

        x = self.bn2(self.project_conv(x))
        if (
            self.block_args.id_skip
            and all(s == 1 for s in self.block_args.strides)
            and self.block_args.input_filters == self.block_args.output_filters
        ):
            x = x + inputs

        return x, total_runtime


# ============================================================================
# singlepath_supernet.py :: SinglePathSuperNet
# ============================================================================


class SinglePathSuperNet(nn.Module):
    def __init__(
        self,
        blocks_args,
        num_classes=1000,
        depth_multiplier=None,
        depth_divisor=8,
        min_depth=None,
        dropout_rate=0.2,
        bn_momentum=0.99,
        bn_eps=1e-3,
        runtime_lut=None,
    ):
        super().__init__()
        runtime_lut = runtime_lut if runtime_lut is not None else _PIXEL1_RUNTIME_LUT

        blocks = []
        block_counter = 0
        for block_args in blocks_args:
            assert block_args.num_repeat > 0
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, depth_multiplier, depth_divisor, min_depth
                ),
                output_filters=round_filters(
                    block_args.output_filters, depth_multiplier, depth_divisor, min_depth
                ),
            )

            layer_runtimes = [
                runtime_lut[str(block_counter)][str(i)]
                for i in range(len(runtime_lut[str(block_counter)]))
            ]
            blocks.append(SinglePathBlock(block_args, layer_runtimes, bn_momentum, bn_eps))
            block_counter += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1]
                )

            for _ in range(block_args.num_repeat - 1):
                layer_runtimes = [
                    runtime_lut[str(block_counter)][str(i)]
                    for i in range(len(runtime_lut[str(block_counter)]))
                ] + [0.7]
                blocks.append(SinglePathBlock(block_args, layer_runtimes, bn_momentum, bn_eps))
                block_counter += 1

        self.blocks = nn.ModuleList(blocks)

        stem_filters = round_filters(32, depth_multiplier, depth_divisor, min_depth)
        self.conv_stem = nn.Conv2d(3, stem_filters, kernel_size=3, stride=2, padding=1, bias=False)
        _conv_kernel_init(self.conv_stem.weight)
        self.bn0 = nn.BatchNorm2d(stem_filters, momentum=1 - bn_momentum, eps=bn_eps)

        self.conv_head = nn.Conv2d(
            blocks_args[-1].output_filters
            if depth_multiplier is None
            else round_filters(
                blocks_args[-1].output_filters, depth_multiplier, depth_divisor, min_depth
            ),
            1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        _conv_kernel_init(self.conv_head.weight)
        self.bn1 = nn.BatchNorm2d(1280, momentum=1 - bn_momentum, eps=bn_eps)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)
        _dense_kernel_init(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate and dropout_rate > 0 else None

    def forward(self, x):
        # rest-of-runtime constant (stem, head, logits, block0, block21) --
        # verbatim from the real repo's `call()`.
        total_runtime = torch.tensor(19.5999, dtype=x.dtype, device=x.device)

        x = F.relu(self.bn0(self.conv_stem(x)))

        for block in self.blocks:
            x, total_runtime = block(x, total_runtime)

        x = F.relu(self.bn1(self.conv_head(x)))
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.fc(x)
        return logits, total_runtime


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_single_path_nas():
    torch.manual_seed(0)
    blocks_args = single_path_search()
    model = SinglePathSuperNet(
        blocks_args,
        num_classes=10,
        depth_multiplier=None,
        dropout_rate=0.2,
    )
    model.eval()
    return model


def example_input_single_path_nas():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    (
        "Single-Path NAS",
        "build_single_path_nas",
        "example_input_single_path_nas",
        2019,
        "ported-pytorch",
    ),
]
