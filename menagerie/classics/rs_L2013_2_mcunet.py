# SOURCE: vendored from https://github.com/mit-han-lab/mcunet @ master
#
# MCUNet (Lin, Chen, Lin, Cohn, Gan, Han. 2020, NeurIPS, "MCUNet: Tiny Deep Learning
# on IoT Devices"). A TinyNAS-searched MobileNet/ProxylessNAS-family architecture
# (`ProxylessNASNets`): a stem conv followed by a sequence of `MobileInvertedResidualBlock`s
# (each an `MBInvertedConvLayer` -- optional 1x1 expand -> depthwise k x k conv -> 1x1
# project -- with an `IdentityLayer` residual shortcut when shapes allow), a global
# average pool, and a linear classifier. The per-block kernel sizes, expansion ratios,
# and channel counts are the output of TinyNAS's constrained architecture search under a
# microcontroller SRAM/Flash budget, not hand-designed. Vendored here is the real
# constructor code from the official MIT HAN Lab repo:
#   https://raw.githubusercontent.com/mit-han-lab/mcunet/master/mcunet/tinynas/nn/networks/proxyless_nets.py
#   https://raw.githubusercontent.com/mit-han-lab/mcunet/master/mcunet/tinynas/nn/modules/layers.py
#   https://raw.githubusercontent.com/mit-han-lab/mcunet/master/mcunet/utils/my_modules.py
#   https://raw.githubusercontent.com/mit-han-lab/mcunet/master/mcunet/utils/pytorch_utils.py
#   https://raw.githubusercontent.com/mit-han-lab/mcunet/master/mcunet/utils/common_tools.py
#   https://raw.githubusercontent.com/mit-han-lab/mcunet/master/mcunet/model_zoo.py
#
# What is kept: every architecture class byte-for-byte (`ConvLayer`, `DepthConvLayer`,
# `PoolingLayer`, `IdentityLayer`, `LinearLayer`, `ZeroLayer`, `MBInvertedConvLayer`,
# `My2DLayer`, `MobileInvertedResidualBlock`, `ProxylessNASNets`, `set_layer_from_config`)
# and their real `build_from_config` wiring, plus the small `MyModule`/`MyNetwork` base
# classes and `build_activation`/`get_same_padding`/`make_divisible` helpers they call.
# The real "mcunet-in2" net config -- the official "mcunet-256kb-1mb_imagenet" model
# (fits 256KB SRAM / 1MB Flash) -- is embedded verbatim below as JSON, fetched from the
# repo's own release asset (`mcunet.model_zoo.build_model('mcunet-in2', pretrained=False)`
# downloads this exact file at
# https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-256kb-1mb_imagenet.json);
# it is embedded here (rather than fetched at trace time) only to avoid a network
# dependency, and describes the real, already-published NAS-discovered 18-block
# architecture (18 `MobileInvertedResidualBlock`s, resolution 160) -- not a synthesized
# or hand-edited config. Random weight init is used (no pretrained checkpoint download).
#
# What is dropped (import plumbing / non-architectural, not architecture): CLI/eval
# scripts (`eval_det.py`, `eval_tflite.py`, `eval_torch.py`), the elastic/dynamic
# supernet training code (`tinynas/elastic_nn/`), the TensorFlow-Lite codebase
# (`tinynas/tf_codebase/`), profiling helpers (`count_net_flops`,
# `count_peak_activation_size`), weight-standardization (`MyConv2d`, unused because the
# real config's `bn.ws_eps` is `null`), and `download_url`/pretrained-checkpoint loading
# (`ProxylessNASNets.build_from_config` never touches pretrained weights; only
# `mcunet.model_zoo.build_model(..., pretrained=True)` does, which we do not call).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import json
import math
from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- mcunet/utils/my_modules.py (MyModule/MyNetwork base classes; trimmed to what
#      ProxylessNASNets/MobileInvertedResidualBlock actually use) ----
class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyNetwork(MyModule):
    CHANNEL_DIVISIBLE = 8


# ---- mcunet/utils/common_tools.py (only the helper actually used by layers.py) ----
def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


# ---- mcunet/utils/pytorch_utils.py (only the activation helpers layers.py uses) ----
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * nn.functional.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu6(x + 3.0, inplace=self.inplace) / 6.0


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
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


# ---- mcunet/utils/pytorch_utils.py (SEModule / ShuffleLayer, referenced by layers.py) ----
class ShuffleLayer(nn.Module):
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


def make_divisible(v, divisor, min_val=None):
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
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


# ---- mcunet/tinynas/nn/modules/layers.py (real architecture classes, byte-faithful) ----
def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


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

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


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
            padding[0] *= self.dilation
            padding[1] *= self.dilation

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

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class DepthConvLayer(My2DLayer):
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

        super(DepthConvLayer, self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order,
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict["depth_conv"] = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.in_channels,
            bias=False,
        )
        weight_dict["point_conv"] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)
        return weight_dict

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)


class PoolingLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_type,
        kernel_size=2,
        stride=2,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        if self.stride == 1:
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == "avg":
            weight_dict["pool"] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == "max":
            weight_dict["pool"] = nn.MaxPool2d(
                self.kernel_size, stride=self.stride, padding=padding
            )
        else:
            raise NotImplementedError
        return weight_dict

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)


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

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


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

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class ZeroLayer(MyModule):
    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        raise ValueError

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)


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

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)


# ---- mcunet/tinynas/nn/networks/proxyless_nets.py (real network wiring) ----
class MobileInvertedResidualBlock(MyModule):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config["mobile_inverted_conv"])
        shortcut = set_layer_from_config(config["shortcut"])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


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

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        feature_mix_layer = set_layer_from_config(config["feature_mix_layer"])
        classifier = set_layer_from_config(config["classifier"])

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if "bn" in config:
            momentum = config["bn"]["momentum"]
            eps = config["bn"]["eps"]
            for m in net.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = momentum
                    m.eps = eps
        return net


# ---- real "mcunet-256kb-1mb_imagenet" net config (mcunet-in2), fetched verbatim from
#      https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-256kb-1mb_imagenet.json
#      -- the actual TinyNAS-searched 18-block architecture, embedded to avoid a
#      network fetch at trace time. Random init (no pretrained checkpoint). ----
_MCUNET_IN2_CONFIG_JSON = r"""

{"name": "ProxylessNASNets", "bn": {"momentum": 0.1, "eps": 1e-05, "ws_eps": null}, "first_conv": {"name": "ConvLayer", "kernel_size": 3, "stride": 2, "dilation": 1, "groups": 1, "bias": false, "has_shuffle": false, "in_channels": 3, "out_channels": 16, "use_bn": true, "act_func": "relu6", "dropout_rate": 0, "ops_order": "weight_bn_act"}, "blocks": [{"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 1, "expand_ratio": 1, "mid_channels": null, "act_func": "relu6", "use_se": false}, "shortcut": null}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 8, "out_channels": 16, "kernel_size": 5, "stride": 2, "expand_ratio": 3, "mid_channels": 24, "act_func": "relu6", "use_se": false}, "shortcut": null}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 16, "out_channels": 16, "kernel_size": 7, "stride": 1, "expand_ratio": 6, "mid_channels": 96, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [16], "out_channels": [16], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "expand_ratio": 5, "mid_channels": 80, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [16], "out_channels": [16], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 16, "out_channels": 16, "kernel_size": 5, "stride": 1, "expand_ratio": 5, "mid_channels": 80, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [16], "out_channels": [16], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 16, "out_channels": 24, "kernel_size": 3, "stride": 2, "expand_ratio": 5, "mid_channels": 80, "act_func": "relu6", "use_se": false}, "shortcut": null}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 24, "out_channels": 24, "kernel_size": 7, "stride": 1, "expand_ratio": 6, "mid_channels": 144, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [24], "out_channels": [24], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 24, "out_channels": 24, "kernel_size": 5, "stride": 1, "expand_ratio": 6, "mid_channels": 144, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [24], "out_channels": [24], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 24, "out_channels": 40, "kernel_size": 7, "stride": 2, "expand_ratio": 4, "mid_channels": 96, "act_func": "relu6", "use_se": false}, "shortcut": null}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 40, "out_channels": 40, "kernel_size": 5, "stride": 1, "expand_ratio": 5, "mid_channels": 200, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [40], "out_channels": [40], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 40, "out_channels": 48, "kernel_size": 3, "stride": 1, "expand_ratio": 5, "mid_channels": 200, "act_func": "relu6", "use_se": false}, "shortcut": null}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 48, "out_channels": 48, "kernel_size": 5, "stride": 1, "expand_ratio": 5, "mid_channels": 240, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [48], "out_channels": [48], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 48, "out_channels": 48, "kernel_size": 3, "stride": 1, "expand_ratio": 4, "mid_channels": 192, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [48], "out_channels": [48], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 48, "out_channels": 96, "kernel_size": 5, "stride": 2, "expand_ratio": 6, "mid_channels": 288, "act_func": "relu6", "use_se": false}, "shortcut": null}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 96, "out_channels": 96, "kernel_size": 5, "stride": 1, "expand_ratio": 4, "mid_channels": 384, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [96], "out_channels": [96], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 96, "out_channels": 96, "kernel_size": 5, "stride": 1, "expand_ratio": 3, "mid_channels": 288, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [96], "out_channels": [96], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 96, "out_channels": 96, "kernel_size": 3, "stride": 1, "expand_ratio": 4, "mid_channels": 384, "act_func": "relu6", "use_se": false}, "shortcut": {"name": "IdentityLayer", "in_channels": [96], "out_channels": [96], "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}}, {"name": "MobileInvertedResidualBlock", "mobile_inverted_conv": {"name": "MBInvertedConvLayer", "in_channels": 96, "out_channels": 160, "kernel_size": 5, "stride": 1, "expand_ratio": 5, "mid_channels": 480, "act_func": "relu6", "use_se": false}, "shortcut": null}], "feature_mix_layer": null, "classifier": {"name": "LinearLayer", "in_features": 160, "out_features": 1000, "bias": true, "use_bn": false, "act_func": null, "dropout_rate": 0, "ops_order": "weight_bn_act"}, "resolution": 160}"""


def build_mcunet_in2():
    # real official "mcunet-256kb-1mb_imagenet" (mcunet-in2) net config; random init.
    config = json.loads(_MCUNET_IN2_CONFIG_JSON)
    return ProxylessNASNets.build_from_config(config)


def example_input_mcunet_in2():
    return torch.randn(1, 3, 160, 160)


MENAGERIE_ENTRIES = [
    ("MCUNet-in2", "build_mcunet_in2", "example_input_mcunet_in2", 2020, "vendored-pytorch"),
]
