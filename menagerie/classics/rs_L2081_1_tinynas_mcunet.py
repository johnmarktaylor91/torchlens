# SOURCE: vendored from mit-han-lab/mcunet @ master
#
# TinyNAS / MCUNet: Tiny Deep Learning on IoT Devices
# Ji Lin, Wei-Ming Chen, Yujun Lin, John Cohn, Chuang Gan, Song Han. NeurIPS 2020.
# https://github.com/mit-han-lab/mcunet
#
# TinyNAS is the two-stage neural-architecture-search method (automated search-space
# optimization + resource-constrained evolutionary search) that MCUNet uses to discover
# tiny CNN backbones for microcontrollers. The architecture family TinyNAS actually
# *discovers and emits* is `ProxylessNASNets` -- a chain of `MobileInvertedResidualBlock`s
# built from `MBInvertedConvLayer` (depthwise-separable inverted-residual blocks with
# optional squeeze-excitation), exactly the OFA/ProxylessNAS supernet building blocks
# vendored here verbatim from `mcunet/tinynas/nn/modules/layers.py`,
# `mcunet/tinynas/nn/networks/proxyless_nets.py`, `mcunet/utils/my_modules.py`, and
# `mcunet/utils/pytorch_modules.py` (only imports/module layout touched -- no
# architectural edits). `mcunet.model_zoo.build_model()` in the real repo instantiates
# this exact class from a downloaded per-net JSON config (the actual TinyNAS search
# output for a given SRAM/flash budget); this staging module builds the same class from
# a small hand-written config in the same schema, shrunk for fast tracing, instead of
# fetching network weights over HTTP.

import math
from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# vendored from mcunet/utils/common_tools.py (get_same_padding)
# ---------------------------------------------------------------------------
def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def make_divisible(v, divisor, min_val=None):
    # vendored from mcunet/utils/pytorch_modules.py
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    # vendored from mcunet/utils/pytorch_modules.py
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
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class ShuffleLayer(nn.Module):
    # vendored from mcunet/utils/pytorch_modules.py
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


class Hswish(nn.Module):
    # vendored from mcunet/utils/pytorch_modules.py
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * nn.functional.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    # vendored from mcunet/utils/pytorch_modules.py
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    # vendored from mcunet/utils/pytorch_modules.py
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()
        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction
        num_mid = make_divisible(
            self.channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE
        )
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


# ---------------------------------------------------------------------------
# vendored from mcunet/utils/my_modules.py (MyModule / MyNetwork base classes)
# ---------------------------------------------------------------------------
class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyNetwork(MyModule):
    CHANNEL_DIVISIBLE = 8


def init_models(net, model_init="he_fout"):
    # vendored from mcunet/utils/my_modules.py
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == "he_fout":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


# ---------------------------------------------------------------------------
# vendored from mcunet/tinynas/nn/modules/layers.py
# ---------------------------------------------------------------------------
class My2DLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        modules["weight"] = self.weight_op()

        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ConvLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding = (padding[0] * self.dilation, padding[1] * self.dilation)

        weight_dict = OrderedDict()
        weight_dict["conv"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)
        return weight_dict


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
                    groups=feature_dim,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SEModule(feature_dim)))
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


class IdentityLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(IdentityLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        return None


class LinearLayer(MyModule):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm1d(in_features)
            else:
                modules["bn"] = nn.BatchNorm1d(out_features)
        else:
            modules["bn"] = None
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        modules["weight"] = {"linear": nn.Linear(self.in_features, self.out_features, self.bias)}

        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


# ---------------------------------------------------------------------------
# vendored from mcunet/tinynas/nn/networks/proxyless_nets.py
# ---------------------------------------------------------------------------
class MobileInvertedResidualBlock(MyModule):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None:
            res = x
        elif self.shortcut is None:
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res


class ProxylessNASNets(MyNetwork):
    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# staging: build a tiny TinyNAS/MCUNet-style ProxylessNASNets directly from real
# vendored classes (the actual net_config a search would emit), instead of the
# HTTP-downloaded config used by mcunet.model_zoo.build_model().
# ---------------------------------------------------------------------------
def build_tinynas_mcunet():
    first_conv = ConvLayer(3, 8, kernel_size=3, stride=2, act_func="relu6")

    block_defs = [
        # (in_c, out_c, kernel, stride, expand_ratio, use_se, has_shortcut)
        (8, 8, 3, 1, 1, False, False),
        (8, 16, 3, 2, 4, False, False),
        (16, 16, 3, 1, 3, True, True),
        (16, 24, 5, 2, 4, True, False),
    ]
    blocks = []
    for in_c, out_c, k, s, e, use_se, has_shortcut in block_defs:
        mb_conv = MBInvertedConvLayer(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            expand_ratio=e,
            use_se=use_se,
        )
        shortcut = IdentityLayer(in_c, out_c) if has_shortcut and in_c == out_c and s == 1 else None
        blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))

    feature_mix_layer = ConvLayer(24, 64, kernel_size=1, stride=1, act_func="relu6")
    classifier = LinearLayer(64, 10, bias=True)

    model = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
    init_models(model)
    model.eval()
    return model


def example_input_tinynas_mcunet():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "TinyNAS (MCUNet ProxylessNASNets backbone)",
        "build_tinynas_mcunet",
        "example_input_tinynas_mcunet",
        2020,
        MENAGERIE_ZOO,
    ),
]
