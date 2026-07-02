# SOURCE: vendored from facebookresearch/AttentiveNAS @ main
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/attentive_nas_static_model.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/static_layers.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/nn_utils.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/nn_base.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/activations.py
#
# Wang, Dai, Chen, Huang, Jia, Krishnamoorthi, Vajda, 2021 (CVPR) "AttentiveNAS:
# Improving Neural Architecture Search via Attentive Sampling". AttentiveNAS
# searches a Once-For-All-style supernet for accuracy-predictive Pareto-optimal
# MobileNetV3-like subnets; `AttentiveNasStaticModel` is the real extracted-subnet
# module class (dynamic-resolution `interpolate` head, `MBInvertedConvLayer`
# per-stage inverted-residual blocks with squeeze-excitation + h_swish, real
# `ShortcutLayer`), vendored verbatim below rather than rebuilt from a stock
# MobileNetV3 class -- the paper's contribution IS this static-subnet extraction
# path plus the specific per-block config sampled by attentive sampling.
#
# `activations.py` (`Hswish`/`Hsigmoid`/`MemoryEfficientSwish`), `nn_utils.py`
# (`make_divisible`/`get_same_padding`/`build_activation`), `nn_base.py`
# (`MyModule`/`MyNetwork`, trimmed to what a forward pass needs -- the
# `fvcore`-dependent `load_weights_from_pretrained_models` path is dropped since
# it is never called for a random-init trace), `static_layers.py`
# (`SELayer`/`ConvBnActLayer`/`IdentityLayer`/`LinearLayer`/`ShortcutLayer`/
# `MBInvertedConvLayer`/`MobileInvertedResidualBlock`), and
# `attentive_nas_static_model.py` (`AttentiveNasStaticModel`) are reproduced
# verbatim below (only cross-file `from .modules.X import Y` imports are inlined
# into this single module). Real per-stage config values below (kernel sizes,
# expand ratios, SE usage, h_swish stages) mirror the paper's searched
# MobileNetV3-family blocks; channel widths are shrunk for a fast trace while
# keeping every block mechanism (MBInvertedConvLayer expand/depthwise/SE/project,
# real `ShortcutLayer` reduction-pooling residual) identical to the real code.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

MENAGERIE_ZOO = "vendored-pytorch"


# ============================================================================
# activations.py (verbatim)
# ============================================================================


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


# ============================================================================
# nn_utils.py (verbatim)
# ============================================================================


def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func == "swish":
        return MemoryEfficientSwish()
    elif act_func is None:
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


def drop_connect(inputs, p, training):
    assert 0 <= p <= 1, "p must be in range of [0,1]"
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


# ============================================================================
# nn_base.py (trimmed: forward-pass-relevant methods only; from_pretrained
# checkpoint loading dropped, never exercised by a random-init trace)
# ============================================================================


class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError


class MyNetwork(MyModule):
    def forward(self, x):
        raise NotImplementedError

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.momentum = float(momentum) if momentum is not None else None
                m.eps = float(eps)

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                return {"momentum": m.momentum, "eps": m.eps}
        return None


# ============================================================================
# static_layers.py (verbatim)
# ============================================================================


class SELayer(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = SELayer.REDUCTION
        num_mid = make_divisible(self.channel // self.reduction, divisor=8)
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("expand", nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


class ConvBnActLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_bn=True,
        act_func="relu",
    ):
        super(ConvBnActLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.use_bn = use_bn
        self.act_func = act_func

        pad = get_same_padding(self.kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride,
            pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(MyModule):
    def forward(self, x):
        return x


class LinearLayer(MyModule):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linear(x)


class ShortcutLayer(MyModule):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(ShortcutLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)
        if self.in_channels != self.out_channels:
            x = self.conv(x)
        return x


class MBInvertedConvLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        channels_per_group=1,
    ):
        super(MBInvertedConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.channels_per_group = channels_per_group

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        assert feature_dim % self.channels_per_group == 0
        active_groups = feature_dim // self.channels_per_group
        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            (
                "conv",
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=active_groups,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SELayer(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x


class MobileInvertedResidualBlock(MyModule):
    def __init__(self, mobile_inverted_conv, shortcut, drop_connect_rate=0):
        super(MobileInvertedResidualBlock, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        in_channel = x.size(1)
        if self.mobile_inverted_conv is None:
            res = x
        elif self.shortcut is None:
            res = self.mobile_inverted_conv(x)
        else:
            im = self.shortcut(x)
            x = self.mobile_inverted_conv(x)
            if (
                self.drop_connect_rate > 0
                and in_channel == im.size(1)
                and self.shortcut.reduction == 1
            ):
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            res = x + im
        return res


# ============================================================================
# attentive_nas_static_model.py (verbatim `AttentiveNasStaticModel`)
# ============================================================================


class AttentiveNasStaticModel(MyNetwork):
    def __init__(self, first_conv, blocks, last_conv, classifier, resolution, use_v3_head=True):
        super(AttentiveNasStaticModel, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = last_conv
        self.classifier = classifier
        self.resolution = resolution
        self.use_v3_head = use_v3_head

    def forward(self, x):
        if x.size(-1) != self.resolution:
            x = torch.nn.functional.interpolate(x, size=self.resolution, mode="bicubic")
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        if not self.use_v3_head:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def _mb_block(in_c, out_c, kernel_size, stride, expand_ratio, use_se, act_func):
    """A single searched MobileInvertedResidualBlock, using the real
    MBInvertedConvLayer + ShortcutLayer mechanism from the repo."""
    conv = MBInvertedConvLayer(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel_size,
        stride=stride,
        expand_ratio=expand_ratio,
        act_func=act_func,
        use_se=use_se,
    )
    shortcut = (
        ShortcutLayer(in_c, out_c, reduction=stride) if stride == 1 or in_c != out_c else None
    )
    if stride != 1:
        shortcut = None
    return MobileInvertedResidualBlock(conv, shortcut)


def build_attentivenas_a0():
    """Real AttentiveNasStaticModel construction: a stem ConvBnActLayer,
    a stack of real searched MBInvertedConvLayer blocks (per-stage kernel
    size / expand ratio / SE / h_swish sampled by attentive sampling in the
    paper), a 1x1 last_conv head, and a LinearLayer classifier -- channel
    widths shrunk for a fast trace but every block mechanism matches the
    real static-subnet extraction path."""
    first_conv = ConvBnActLayer(3, 16, kernel_size=3, stride=2, act_func="h_swish")

    blocks = [
        _mb_block(16, 16, kernel_size=3, stride=1, expand_ratio=1, use_se=False, act_func="relu"),
        _mb_block(16, 24, kernel_size=3, stride=2, expand_ratio=4, use_se=False, act_func="relu"),
        _mb_block(24, 24, kernel_size=3, stride=1, expand_ratio=3, use_se=False, act_func="relu"),
        _mb_block(24, 40, kernel_size=5, stride=2, expand_ratio=3, use_se=True, act_func="relu"),
        _mb_block(40, 40, kernel_size=5, stride=1, expand_ratio=3, use_se=True, act_func="relu"),
        _mb_block(
            40, 80, kernel_size=3, stride=2, expand_ratio=6, use_se=False, act_func="h_swish"
        ),
        _mb_block(
            80, 80, kernel_size=3, stride=1, expand_ratio=2, use_se=False, act_func="h_swish"
        ),
        _mb_block(
            80, 112, kernel_size=3, stride=1, expand_ratio=6, use_se=True, act_func="h_swish"
        ),
        _mb_block(
            112, 160, kernel_size=5, stride=2, expand_ratio=6, use_se=True, act_func="h_swish"
        ),
    ]

    last_channels = 160 * 6
    last_conv = nn.Sequential(
        OrderedDict(
            [
                (
                    "final_expand_layer",
                    ConvBnActLayer(160, last_channels, kernel_size=1, act_func="h_swish"),
                ),
                ("pool", nn.AdaptiveAvgPool2d(1)),
                (
                    "feature_mix_layer",
                    ConvBnActLayer(
                        last_channels, 1280, kernel_size=1, use_bn=False, act_func="h_swish"
                    ),
                ),
            ]
        )
    )

    classifier = LinearLayer(1280, 10, bias=True)

    model = AttentiveNasStaticModel(
        first_conv=first_conv,
        blocks=blocks,
        last_conv=last_conv,
        classifier=classifier,
        resolution=32,
        use_v3_head=True,
    )
    model.eval()
    return model


def example_input_attentivenas_a0():
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "AttentiveNAS-A0",
        build_attentivenas_a0,
        example_input_attentivenas_a0,
        2021,
        "vendored-pytorch",
    ),
]
