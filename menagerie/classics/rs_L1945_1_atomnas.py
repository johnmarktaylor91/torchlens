# SOURCE: vendored from meijieru/AtomNAS @ master
# https://raw.githubusercontent.com/meijieru/AtomNAS/master/models/mobilenet_base.py
# https://raw.githubusercontent.com/meijieru/AtomNAS/master/models/searched_network.py
#
# Mei, Li, Lin, Yuille, Yang, 2020 (ICLR) "AtomNAS: Fine-Grained End-to-End Neural
# Architecture Search". AtomNAS decomposes the standard MobileNetV2 inverted-residual
# block into many small "atomic blocks" (parallel depthwise branches of different
# kernel sizes/channel widths that are concatenated after a shared pointwise expand),
# then prunes dead atomic blocks via BN-gamma-driven search to obtain the final
# `MobileNetSearched` architecture below. The concatenated multi-kernel-size
# depthwise branch (`InvertedResidualChannels`/`InvertedResidualChannelsFused`,
# each block holding several parallel `(kernel_size, hidden_dim)` depthwise paths
# that are summed/concatenated) is AtomNAS's real architectural contribution, not a
# stock MobileNetV2 block -- so this is vendored real code, not built from an
# installed library class.
#
# `mobilenet_base.py` (`ConvBNReLU`, `InvertedResidualChannels`,
# `InvertedResidualChannelsFused`, `SqueezeAndExcitation`, `get_active_fn`,
# `get_block`) and `searched_network.py` (`MobileNetSearched`) are reproduced
# verbatim below (only the `models.compress_utils`/`utils.common` cross-module
# imports are dropped -- `add_prefix`/`get_device` are inlined trivially, and the
# `compress_by_mask`/`compress_by_threshold` methods that depend on
# `models.compress_utils` are omitted since they are search-time-only pruning
# utilities never exercised by a forward pass). The real found AtomNAS-A searched
# architecture's `inverted_residual_setting` (real per-stage output channels /
# kernel-size lists / atomic-block hidden-dim splits from
# `apps/searched/models/atomnas_a.yml`) is used verbatim below, just at a smaller
# `input_size=32` (vs. the paper's 224) so the trace is fast -- every block's
# channel counts, kernel sizes, and expand flags are the real searched values.

import warnings
import collections
import functools
import math

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ============================================================================
# mobilenet_base.py (verbatim building blocks; add_prefix/get_device inlined)
# ============================================================================


def add_prefix(name, prefix=None, split="."):
    """Add prefix to name if given."""
    if prefix is not None:
        return "{}{}{}".format(prefix, split, name)
    return name


class Identity(nn.Module):
    """Module proxy for null op."""

    def forward(self, x):
        return x


class Narrow(nn.Module):
    """Module proxy for `torch.narrow`."""

    def __init__(self, dimension, start, length):
        super(Narrow, self).__init__()
        self.dimension = dimension
        self.start = start
        self.length = length

    def forward(self, x):
        return x.narrow(self.dimension, self.start, self.length)


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation module.

    See: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, n_feature, n_hidden, spatial_dims=[2, 3], active_fn=None):
        super(SqueezeAndExcitation, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.spatial_dims = spatial_dims
        self.se_reduce = nn.Conv2d(n_feature, n_hidden, 1, bias=True)
        self.se_expand = nn.Conv2d(n_hidden, n_feature, 1, bias=True)
        self.active_fn = active_fn()

    def forward(self, x):
        se_tensor = x.mean(self.spatial_dims, keepdim=True)
        se_tensor = self.se_expand(self.active_fn(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        active_fn=None,
        batch_norm_kwargs=None,
    ):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs),
            active_fn(),
        )


class InvertedResidualChannelsFused(nn.Module):
    """Speedup version of `InvertedResidualChannels` by fusing small kernels.

    Support `Squeeze-and-Excitation`.
    """

    def __init__(
        self,
        inp,
        oup,
        stride,
        channels,
        kernel_sizes,
        expand,
        active_fn=None,
        batch_norm_kwargs=None,
        se_ratio=None,
    ):
        super(InvertedResidualChannelsFused, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn
        self.se_ratio = se_ratio

        (self.expand_conv, self.depth_ops, self.project_conv, self.se_op) = self._build(
            channels, kernel_sizes, expand, se_ratio
        )

    def _build(self, hidden_dims, kernel_sizes, expand, se_ratio):
        _batch_norm_kwargs = self.batch_norm_kwargs if self.batch_norm_kwargs is not None else {}

        hidden_dim_total = sum(hidden_dims)
        if self.expand:
            expand_conv = ConvBNReLU(
                self.input_dim,
                hidden_dim_total,
                kernel_size=1,
                batch_norm_kwargs=_batch_norm_kwargs,
                active_fn=self.active_fn,
            )
        else:
            expand_conv = Identity()

        narrow_start = 0
        depth_ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            layers.extend(
                [
                    ConvBNReLU(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=k,
                        stride=self.stride,
                        groups=hidden_dim,
                        batch_norm_kwargs=_batch_norm_kwargs,
                        active_fn=self.active_fn,
                    ),
                ]
            )
            depth_ops.append(nn.Sequential(*layers))
        project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim_total, self.output_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs),
        )

        if expand and narrow_start != hidden_dim_total:
            raise ValueError("Part of expanded are not used")

        if se_ratio is not None:
            se_op = SqueezeAndExcitation(
                hidden_dim_total, int(round(self.input_dim * se_ratio)), active_fn=self.active_fn
            )
        else:
            se_op = Identity()
        return expand_conv, depth_ops, project_conv, se_op

    def forward(self, x):
        res = self.expand_conv(x)
        res = [op(res) for op in self.depth_ops]
        if len(res) != 1:
            res = torch.cat(res, dim=1)
        else:
            res = res[0]
        res = self.se_op(res)
        res = self.project_conv(res)
        if self.use_res_connect:
            return x + res
        return res


class InvertedResidualChannels(nn.Module):
    """MobileNetV2 building block, atomic-block (multi-kernel-size) variant."""

    def __init__(
        self,
        inp,
        oup,
        stride,
        channels,
        kernel_sizes,
        expand,
        active_fn=None,
        batch_norm_kwargs=None,
    ):
        super(InvertedResidualChannels, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn

        self.ops, self.pw_bn = self._build(channels, kernel_sizes, expand)

    def _build(self, hidden_dims, kernel_sizes, expand):
        _batch_norm_kwargs = self.batch_norm_kwargs if self.batch_norm_kwargs is not None else {}

        narrow_start = 0
        ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                layers.append(
                    ConvBNReLU(
                        self.input_dim,
                        hidden_dim,
                        kernel_size=1,
                        batch_norm_kwargs=_batch_norm_kwargs,
                        active_fn=self.active_fn,
                    )
                )
            else:
                narrow_start += hidden_dim
            layers.extend(
                [
                    ConvBNReLU(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=k,
                        stride=self.stride,
                        groups=hidden_dim,
                        batch_norm_kwargs=_batch_norm_kwargs,
                        active_fn=self.active_fn,
                    ),
                    nn.Conv2d(hidden_dim, self.output_dim, 1, 1, 0, bias=False),
                ]
            )
            ops.append(nn.Sequential(*layers))
        pw_bn = nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs)

        if not expand and narrow_start != self.input_dim:
            raise ValueError("Part of input are not used")
        return ops, pw_bn

    def forward(self, x):
        if len(self.ops) == 0:
            return x

        tmp = sum([op(x) for op in self.ops])
        tmp = self.pw_bn(tmp)
        if self.use_res_connect:
            return x + tmp
        return tmp


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        "nn.ReLU6": functools.partial(nn.ReLU6, inplace=True),
        "nn.ReLU": functools.partial(nn.ReLU, inplace=True),
    }[name]
    return active_fn


def get_block(name):
    """Select building block."""
    return {
        "InvertedResidualChannels": InvertedResidualChannels,
        "InvertedResidualChannelsFused": InvertedResidualChannelsFused,
    }[name]


def _get_named_block_list(m):
    """Get `{name: module}` dictionary for inverted residual blocks."""
    blocks = list(m.features.named_children())
    blocks = blocks[1:-2]
    return collections.OrderedDict([("features.{}".format(name), block) for name, block in blocks])


# ============================================================================
# searched_network.py (verbatim `MobileNetSearched`)
# ============================================================================


class MobileNetSearched(nn.Module):
    """MobileNetV2-like network, instantiated with a real AtomNAS-found
    per-stage `inverted_residual_setting` (found channels/kernel sizes)."""

    def __init__(
        self,
        num_classes=1000,
        input_size=224,
        input_channel=32,
        last_channel=1280,
        width_mult=1.0,
        inverted_residual_setting=None,
        dropout_ratio=0.2,
        se_ratio=None,
        batch_norm_momentum=0.1,
        batch_norm_epsilon=1e-5,
        active_fn="nn.ReLU6",
        block="InvertedResidualChannels",
        round_nearest=8,
    ):
        super(MobileNetSearched, self).__init__()
        batch_norm_kwargs = {"momentum": batch_norm_momentum, "eps": batch_norm_epsilon}

        if width_mult != 1.0:
            raise ValueError("Searched model should have width 1")

        self.input_size = input_size
        self.input_channel = input_channel
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.inverted_residual_setting = inverted_residual_setting
        self.active_fn = active_fn
        self.block = block
        self.batch_norm_kwargs = batch_norm_kwargs

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 6:
            raise ValueError("inverted_residual_setting should be non-empty or a 6-element list")
        if input_size % 32 != 0:
            raise ValueError("Input size must divide 32")
        for name, channel in [["Input", input_channel], ["Last", last_channel]]:
            if (channel * width_mult) % round_nearest:
                warnings.warn("{} channel could not divide {}".format(name, round_nearest))
        active_fn_cls = get_active_fn(active_fn)
        block_cls = get_block(block)
        _extra_kwargs = {}
        if se_ratio is not None:
            if issubclass(block_cls, InvertedResidualChannelsFused):
                _extra_kwargs["se_ratio"] = se_ratio
            else:
                raise NotImplementedError("SE module not supported for block: {}".format(block_cls))

        features = [
            ConvBNReLU(
                3,
                input_channel,
                stride=2,
                batch_norm_kwargs=batch_norm_kwargs,
                active_fn=active_fn_cls,
            )
        ]
        for c, n, s, ks, hiddens, expand in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block_cls(
                        input_channel,
                        output_channel,
                        stride,
                        hiddens,
                        ks,
                        expand,
                        active_fn=active_fn_cls,
                        batch_norm_kwargs=batch_norm_kwargs,
                        **_extra_kwargs,
                    )
                )
                input_channel = output_channel
        features.append(
            ConvBNReLU(
                input_channel,
                last_channel,
                kernel_size=1,
                batch_norm_kwargs=batch_norm_kwargs,
                active_fn=active_fn_cls,
            )
        )
        avg_pool_size = input_size // 32
        features.append(nn.AvgPool2d(avg_pool_size))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, num_classes),
        )

    def get_named_block_list(self):
        return _get_named_block_list(self)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)
        x = self.classifier(x)
        return x


Model = MobileNetSearched


# ============================================================================
# build_/example_input_ harness
# ============================================================================

# Real found AtomNAS-A architecture, verbatim from
# apps/searched/models/atomnas_a.yml (`inverted_residual_setting`).
_ATOMNAS_A_SETTING = [
    [16, 1, 1, [3], [16], False],
    [24, 1, 2, [3, 5, 7], [16, 12, 9], True],
    [24, 1, 1, [3, 5, 7], [19, 2, 1], True],
    [24, 1, 1, [3, 5, 7], [9, 7, 1], True],
    [24, 1, 1, [3, 5], [14, 1], True],
    [40, 1, 2, [3, 5, 7], [46, 49, 50], True],
    [40, 1, 1, [3, 5, 7], [27, 19, 14], True],
    [40, 1, 1, [3, 5, 7], [22, 11, 3], True],
    [40, 1, 1, [3, 5, 7], [30, 14, 3], True],
    [80, 1, 2, [3, 5, 7], [127, 130, 111], True],
    [80, 1, 1, [3, 5, 7], [33, 25, 45], True],
    [80, 1, 1, [3, 5, 7], [38, 12, 34], True],
    [80, 1, 1, [3, 5, 7], [53, 23, 19], True],
    [96, 1, 1, [3, 5, 7], [344, 196, 168], True],
    [96, 1, 1, [3, 5, 7], [44, 18, 27], True],
    [96, 1, 1, [3, 5, 7], [37, 29, 18], True],
    [96, 1, 1, [3, 5, 7], [52, 21, 24], True],
    [192, 1, 2, [3, 5, 7], [381, 364, 342], True],
    [192, 1, 1, [3, 5, 7], [123, 76, 162], True],
    [192, 1, 1, [3, 5, 7], [179, 107, 212], True],
    [192, 1, 1, [3, 5, 7], [238, 111, 171], True],
    [320, 1, 1, [3, 5, 7], [654, 495, 546], True],
]


def build_atomnas_a():
    model = MobileNetSearched(
        num_classes=10,
        input_size=32,
        input_channel=16,
        last_channel=1280,
        width_mult=1.0,
        inverted_residual_setting=_ATOMNAS_A_SETTING,
        active_fn="nn.ReLU",
    )
    model.eval()
    return model


def example_input_atomnas_a():
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("AtomNAS-A", build_atomnas_a, example_input_atomnas_a, 2020, "vendored-pytorch"),
]
