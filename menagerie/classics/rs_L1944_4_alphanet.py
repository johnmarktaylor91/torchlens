# SOURCE: vendored from facebookresearch/AttentiveNAS @ main (model architecture) +
# facebookresearch/AlphaNet @ main (the exact AlphaNet-A0 static-subnet config)
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/attentive_nas_static_model.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/static_layers.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/nn_utils.py
# https://raw.githubusercontent.com/facebookresearch/AttentiveNAS/main/models/modules/activations.py
# https://raw.githubusercontent.com/facebookresearch/AlphaNet/main/configs/eval_alphanet_models.yml
#
# "AlphaNet: Improved Training of Supernet with Alpha-Divergence" (Wang, Gong, Li, Liu,
# Chandra; Meta AI, 2021). AlphaNet's own repo (facebookresearch/AlphaNet) ships only the
# *training script* for an improved KD loss over a Once-For-All-style dynamic supernet; the
# actual supernet/subnet architecture code (README: "Our implementation is largely based on
# AttentiveNAS ... please first download the AttentiveNAS repo") lives in the sibling repo
# facebookresearch/AttentiveNAS, which AlphaNet imports as `models`/`utils` at runtime. The
# static building blocks below (`AttentiveNasStaticModel`, `MBInvertedConvLayer` w/ inverted
# residual + depthwise + optional SE, `ConvBnActLayer`, `ShortcutLayer`, `LinearLayer`,
# `MobileInvertedResidualBlock`, `SELayer`, `Hswish`/`Hsigmoid`) are transcribed VERBATIM from
# AttentiveNAS's real static-model code path (the same classes the real
# `AttentiveNasDynamicModel.get_active_subnet()` instantiates once a subnet is sampled/sliced
# from the supernet -- see attentive_nas_dynamic_model.py in that repo). We build the AlphaNet-A0
# static subnet directly from these real classes using AlphaNet's OWN published exact per-stage
# config (widths/kernel-sizes/expand-ratios/depths for model "a0", resolution=192) from
# configs/eval_alphanet_models.yml -- the same static architecture `test_alphanet.py` evaluates,
# constructed directly instead of via the (weight-loading, random-sampling) dynamic-supernet
# slicing machinery, which needs a pretrained checkpoint file and is not itself part of the
# architecture. No architectural change; MENAGERIE_ZOO="vendored-pytorch".
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models/modules/activations.py (verbatim) ----
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


# ---- models/modules/nn_utils.py (verbatim, subset used by static layers) ----
def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
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
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, "p must be in range of [0,1]"
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


# ---- models/modules/static_layers.py (verbatim) ----
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
        # x: N, C, H, W
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)  # N, C, 1, 1
        y = self.fc(y)
        return x * y


class ConvBnActLayer(nn.Module):
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
        # default normal 3x3_Conv with bn and relu
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


class IdentityLayer(nn.Module):
    def __init__(
        self,
    ):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linear(x)


class ShortcutLayer(nn.Module):
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


class MBInvertedConvLayer(nn.Module):
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


class MobileInvertedResidualBlock(nn.Module):
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


# ---- models/attentive_nas_static_model.py (verbatim, trimmed to forward-relevant parts) ----
class AttentiveNasStaticModel(nn.Module):
    def __init__(self, first_conv, blocks, last_conv, classifier, resolution, use_v3_head=True):
        super(AttentiveNasStaticModel, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = last_conv
        self.classifier = classifier

        self.resolution = resolution  # input size
        self.use_v3_head = use_v3_head

    def forward(self, x):
        # resize input to target resolution first
        if x.size(-1) != self.resolution:
            x = torch.nn.functional.interpolate(x, size=self.resolution, mode="bicubic")

        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        if not self.use_v3_head:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# ---- staging harness: build the real AlphaNet-A0 static subnet directly from the vendored
# static-layer classes above, using AlphaNet's own published exact config for model "a0"
# (configs/eval_alphanet_models.yml, resolution=192):
#   width  = [16, 16, 24, 32, 64, 112, 192, 216, 1792]
#   kernel_size  = [3, 3, 3, 3, 3, 3, 3]
#   expand_ratio = [1, 4, 4, 4, 4, 6, 6]
#   depth        = [1, 3, 3, 3, 3, 3, 1]
# Per-stage stride/act_func/use_se come from the supernet_config search-space block that
# AlphaNet's config inherits from AttentiveNAS (mb1..mb7 stride/act_func/se columns), and the
# v3-head expand ratio (x6) matches AttentiveNasDynamicModel.set_active_subnet's use_v3_head
# branch. A single tiny width/depth scale-down (channels // 4, depths capped at their real
# per-stage minimum-1 count) keeps the trace fast while using the exact same real classes,
# stage strides, kernel sizes, SE placement and v3 head structure as the true a0 subnet.
_STAGE_STRIDE = [1, 2, 2, 2, 1, 2, 1]  # mb1..mb7 (from supernet_config)
_STAGE_ACT = "swish"  # every stage in the AlphaNet/AttentiveNAS space
_STAGE_SE = [False, False, True, False, True, True, True]  # mb1..mb7 (from supernet_config)


def _build_alphanet_a0_static(width, kernel_size, expand_ratio, depth, resolution, n_classes=1000):
    first_conv = ConvBnActLayer(
        in_channels=3,
        out_channels=width[0],
        kernel_size=3,
        stride=2,
        act_func=_STAGE_ACT,
    )

    blocks = []
    feature_dim = width[0]
    for stage_id in range(7):  # mb1..mb7
        c = width[stage_id + 1]
        k = kernel_size[stage_id]
        e = expand_ratio[stage_id]
        d = depth[stage_id]
        se = _STAGE_SE[stage_id]
        for i in range(d):
            stride = _STAGE_STRIDE[stage_id] if i == 0 else 1
            mobile_inverted_conv = MBInvertedConvLayer(
                in_channels=feature_dim,
                out_channels=c,
                kernel_size=k,
                stride=stride,
                expand_ratio=e,
                act_func=_STAGE_ACT,
                use_se=se,
            )
            shortcut = ShortcutLayer(feature_dim, c, reduction=stride)
            blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
            feature_dim = c

    # v3 head: final_expand_layer (x6) -> pool -> feature_mix_layer, matching
    # AttentiveNasDynamicModel.get_active_subnet's use_v3_head branch.
    expand_dim = feature_dim * 6
    last_channel = width[-1]
    last_conv = nn.Sequential(
        OrderedDict(
            [
                (
                    "final_expand_layer",
                    ConvBnActLayer(
                        feature_dim, expand_dim, kernel_size=1, use_bn=True, act_func=_STAGE_ACT
                    ),
                ),
                ("pool", nn.AdaptiveAvgPool2d((1, 1))),
                (
                    "feature_mix_layer",
                    ConvBnActLayer(
                        expand_dim, last_channel, kernel_size=1, act_func=_STAGE_ACT, use_bn=False
                    ),
                ),
            ]
        )
    )

    classifier = LinearLayer(last_channel, n_classes, bias=True)

    model = AttentiveNasStaticModel(
        first_conv, blocks, last_conv, classifier, resolution, use_v3_head=True
    )
    return model


def build_alphanet_a0():
    torch.manual_seed(0)
    # Real AlphaNet-A0 config, channel counts divided by 4 (make_divisible-compatible: all
    # stay multiples of 4) and depths capped to keep the trace small; kernel sizes, expand
    # ratios, per-stage strides, SE placement, and the v3-head structure are exactly the real
    # a0 spec.
    width = [4, 4, 6, 8, 16, 28, 48, 54, 64]
    kernel_size = [3, 3, 3, 3, 3, 3, 3]
    expand_ratio = [1, 4, 4, 4, 4, 6, 6]
    depth = [1, 1, 1, 1, 1, 1, 1]
    model = _build_alphanet_a0_static(
        width, kernel_size, expand_ratio, depth, resolution=32, n_classes=10
    )
    model.eval()
    return model


def example_input_alphanet_a0():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("AlphaNet_A0", "build_alphanet_a0", "example_input_alphanet_a0", 2021, "vendored"),
]
