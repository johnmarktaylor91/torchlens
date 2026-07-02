# SOURCE: vendored from xfey/pytorch-BigNAS @ master
# https://raw.githubusercontent.com/xfey/pytorch-BigNAS/master/bignas/utils.py
# https://raw.githubusercontent.com/xfey/pytorch-BigNAS/master/bignas/ops.py
# https://raw.githubusercontent.com/xfey/pytorch-BigNAS/master/bignas/dynamic_ops.py
# https://raw.githubusercontent.com/xfey/pytorch-BigNAS/master/bignas/dynamic_layers.py
# https://raw.githubusercontent.com/xfey/pytorch-BigNAS/master/bignas/cnn.py
#
# "BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models"
# (Yu et al., ECCV 2020). Community PyTorch reimplementation by xfey (adapted from
# Facebook's AttentiveNAS and the XNAS repo -- no official Google code release).
# The classes below (DynamicConv2d/DynamicBatchNorm2d/DynamicSeparableConv2d/DynamicSE,
# DynamicConvLayer/DynamicMBConvLayer/DynamicShortcutLayer/DynamicLinearLayer,
# BigNASDynamicModel/BigNASStaticModel, and the static ConvLayer/MBConvLayer/
# ShortcutLayer/LinearLayer/SEModule/ResidualBlock building blocks) are transcribed
# verbatim from the four real repo files above -- only the `from bignas.xxx import yyy`
# module-path imports were flattened to local references (everything now lives in one
# file, in the same dependency order: utils -> ops -> dynamic_ops -> dynamic_layers ->
# cnn) and the `import torch.distributed as dist` sync-batchnorm branch is untouched
# (it is simply never exercised at world_size=1 / not self.need_sync). No architecture
# was changed. BigNASDynamicModel is the elastic supernet (slimmable width/depth/kernel/
# expand-ratio search space, trained via progressive shrinking); calling its
# `get_active_subnet()` extracts a single concrete sub-architecture as a static
# BigNASStaticModel with ordinary nn.Conv2d/nn.BatchNorm2d layers -- exactly the
# real "sampled subnet" entry point the repo itself uses for evaluation/deployment.
import copy
import math
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

MENAGERIE_ZOO = "vendored-pytorch"


# ---- bignas/utils.py (verbatim) ----

multiply_adds = 1


def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x):
    return list_sum(x) / len(x)


def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


# ---- bignas/ops.py (verbatim) ----


def make_divisible(v, divisor=8, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


"""Activation."""


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
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


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

    def __repr__(self):
        return "Hswish()"


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


class WeightStandardConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(WeightStandardConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.WS_EPS
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(WeightStandardConv2d, self).forward(x)
        else:
            return F.conv2d(
                x,
                self.weight_standardization(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        return super(WeightStandardConv2d, self).__repr__()[:-1] + ", ws_eps=%s)" % self.WS_EPS


# Basic layer to init with Dropout, Conv, BN and activations (order not fixed.)
class MixedDCBRLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(MixedDCBRLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act" and self.use_bn)
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
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
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def __repr__(self):
        return "ShuffleLayer(groups=%d)" % self.groups


class ConvLayer(MixedDCBRLayer):
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
        use_se=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_se = use_se

        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )
        if self.use_se:
            self.add_module("se", SEModule(self.out_channels))

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )
            }
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        if self.use_se:
            conv_str = "SE_" + conv_str
        conv_str += "_" + self.act_func.upper()
        if self.use_bn:
            if isinstance(self.bn, nn.GroupNorm):
                conv_str += "_GN%d" % self.bn.num_groups
            elif isinstance(self.bn, nn.BatchNorm2d):
                conv_str += "_BN"
        return conv_str

    @property
    def config(self):
        return {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
            "use_se": self.use_se,
            **super(ConvLayer, self).config,
        }


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self.channel // self.reduction)

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

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


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


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return "Zero"

    @property
    def config(self):
        return {
            "name": ZeroLayer.__name__,
        }


class ResidualBlock(nn.Module):
    def __init__(self, conv, shortcut, drop_connect_rate=0):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.mobile_inverted_conv = self.conv  # BigNAS
        self.shortcut = shortcut
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        in_channel = x.size(1)
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x)
        else:
            im = self.shortcut(x)
            x = self.conv(x)
            if (
                self.drop_connect_rate > 0
                and in_channel == im.size(1)
                and self.shortcut.reduction == 1
            ):
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            res = x + im
        return res

    @property
    def module_str(self):
        return "(%s, %s)" % (
            self.conv.module_str if self.conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None,
        )

    @property
    def config(self):
        return {
            "name": ResidualBlock.__name__,
            "conv": self.conv.config if self.conv is not None else None,
            "shortcut": self.shortcut.config if self.shortcut is not None else None,
        }

    @property
    def mobile_inverted_conv(self):
        return self.conv


class MBConvLayer(nn.Module):
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
        super(MBConvLayer, self).__init__()

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

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        if self.channels_per_group is not None:
            layer_str += "_G%d" % self.channels_per_group
        if isinstance(self.point_linear.bn, nn.GroupNorm):
            layer_str += "_GN%d" % self.point_linear.bn.num_groups
        elif isinstance(self.point_linear.bn, nn.BatchNorm2d):
            layer_str += "_BN"

        return layer_str

    @property
    def config(self):
        return {
            "name": MBConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "channeles_per_group": self.channels_per_group,
        }


class LinearLayer(nn.Module):
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

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm1d(in_features)
            else:
                modules["bn"] = nn.BatchNorm1d(out_features)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # linear
        modules["weight"] = {"linear": nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
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

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class ShortcutLayer(nn.Module):
    """
    NOTE:
    This class implements similar functionality to `IdentityLayer`,
    but adds and removes part of the implementation.
    """

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

    @property
    def module_str(self):
        if self.in_channels == self.out_channels and self.reduction == 1:
            conv_str = "IdentityShortcut"
        else:
            if self.reduction == 1:
                conv_str = "%d-%d_Shortcut" % (self.in_channels, self.out_channels)
            else:
                conv_str = "%d-%d_R%d_Shortcut" % (
                    self.in_channels,
                    self.out_channels,
                    self.reduction,
                )
        return conv_str

    @property
    def config(self):
        return {
            "name": ShortcutLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "reduction": self.reduction,
        }

    @staticmethod
    def build_from_config(config):
        return ShortcutLayer(**config)


# ---- bignas/dynamic_ops.py (verbatim) ----


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels if isinstance(target_bn, nn.GroupNorm) else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y


class DynamicSeparableConv2d(nn.Module):
    def __init__(
        self, max_in_channels, kernel_size_list, stride=1, dilation=1, channels_per_group=1
    ):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.channels_per_group = channels_per_group
        assert self.max_in_channels % self.channels_per_group == 0
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels // self.channels_per_group,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        assert in_channel % self.channels_per_group == 0

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            in_channel // self.channels_per_group,
        )
        return y


class DynamicConv2d(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, WeightStandardConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.reduce.weight[:num_mid, :in_channel, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(self.fc.reduce.weight[:num_mid, :, :, :], groups, dim=1)
            return torch.cat(
                [sub_filter[:, :sub_in_channels, :, :] for sub_filter in sub_filters],
                dim=1,
            )

    def get_active_reduce_bias(self, num_mid):
        return self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None

    def get_active_expand_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.weight[:in_channel, :num_mid, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(self.fc.expand.weight[:, :num_mid, :, :], groups, dim=0)
            return torch.cat(
                [sub_filter[:sub_in_channels, :, :, :] for sub_filter in sub_filters],
                dim=0,
            )

    def get_active_expand_bias(self, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.bias[:in_channel] if self.fc.expand.bias is not None else None
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            return torch.cat([sub_bias[:sub_in_channels] for sub_bias in sub_bias_list], dim=0)

    def forward(self, x, groups=None):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel, groups=groups)
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicBatchNorm2d(nn.Module):
    """
    1. doesn't acculate bn statistics, (momentum=0.)
    2. calculate BN statistics of all subnets after training
    3. bn weights are shared
    https://arxiv.org/abs/1903.05134
    https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
    """

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

        self.need_sync = False  # sync-batchnormalization, suggested to use in bignas

        # reserved to tracking the performance of the largest and smallest network
        self.bn_tracking = nn.ModuleList(
            [
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
            ]
        )

    def forward(self, x):
        feature_dim = x.size(1)
        if not self.training:
            raise ValueError("DynamicBN only supports training")

        bn = self.bn
        # need_sync
        if not self.need_sync:
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                bn.momentum,
                bn.eps,
            )
        else:
            assert dist.get_world_size() > 1, "SyncBatchNorm requires >1 world size"
            B, C = x.shape[0], x.shape[1]
            mean = torch.mean(x, dim=[0, 2, 3])
            meansqr = torch.mean(x * x, dim=[0, 2, 3])
            assert B > 0, "does not support zero batch size"
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)

            var = meansqr - mean * mean
            invstd = torch.rsqrt(var + bn.eps)
            scale = bn.weight[:feature_dim] * invstd
            bias = bn.bias[:feature_dim] - mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias


# ---- bignas/dynamic_layers.py (verbatim) ----


class DynamicLinearLayer(nn.Module):
    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "DyLinear(%d, %d)" % (max(self.in_features_list), self.out_features)

    @property
    def config(self):
        return {
            "name": DynamicLinear.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = LinearLayer(
            in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate
        )
        sub_layer = sub_layer.to(self.parameters().__next__().device)
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.get_active_weight(self.out_features, in_features).data
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(self.linear.get_active_bias(self.out_features).data)
        return sub_layer

    def get_active_subnet_config(self, in_features):
        return {
            "name": LinearLayer.__name__,
            "in_features": in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
        }


class DynamicMBConvLayer(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride=1,
        act_func="relu6",
        use_se=False,
        channels_per_group=1,
    ):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = val2list(in_channel_list)
        self.out_channel_list = val2list(out_channel_list)

        self.kernel_size_list = val2list(kernel_size_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.channels_per_group = channels_per_group

        # build modules
        max_middle_channel = round(max(self.in_channel_list) * max(self.expand_ratio_list))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicSeparableConv2d(
                            max_middle_channel,
                            self.kernel_size_list,
                            stride=self.stride,
                            channels_per_group=self.channels_per_group,
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        if self.use_se:
            self.depth_conv.add_module("se", DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio), 8
            )

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return "SE(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )
        else:
            return "(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )

    @property
    def config(self):
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "channels_per_group": self.channels_per_group,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)
        channels_per_group = self.depth_conv.conv.channels_per_group

        # build the new layer
        sub_layer = MBConvLayer(
            in_channel,
            self.active_out_channel,
            self.active_kernel_size,
            self.stride,
            self.active_expand_ratio,
            act_func=self.act_func,
            mid_channels=middle_channel,
            use_se=self.use_se,
            channels_per_group=channels_per_group,
        )
        sub_layer = sub_layer.to(self.parameters().__next__().device)

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[:middle_channel, :in_channel, :, :]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.fc.reduce.bias.data[:se_mid]
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.fc.expand.bias.data[:middle_channel]
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[
                : self.active_out_channel, :middle_channel, :, :
            ]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError


class DynamicConvLayer(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = val2list(in_channel_list)
        self.out_channel_list = val2list(out_channel_list)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))

        if self.act_func is not None:
            self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (self.active_out_channel, self.kernel_size, self.stride)

    @property
    def config(self):
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ConvLayer(
            in_channel,
            self.active_out_channel,
            self.kernel_size,
            self.stride,
            self.dilation,
            use_bn=self.use_bn,
            act_func=self.act_func,
        )
        sub_layer = sub_layer.to(self.parameters().__next__().device)

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :]
        )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer


class DynamicShortcutLayer(nn.Module):
    def __init__(self, in_channel_list, out_channel_list, reduction=1):
        super(DynamicShortcutLayer, self).__init__()

        self.in_channel_list = val2list(in_channel_list)
        self.out_channel_list = val2list(out_channel_list)
        self.reduction = reduction

        self.conv = DynamicConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=1,
            stride=1,
        )

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        # identity mapping
        if in_channel == self.active_out_channel and self.reduction == 1:
            return x
        # average pooling, if size doesn't match
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)

        # 1*1 conv, if #channels doesn't match
        if in_channel != self.active_out_channel:
            self.conv.active_out_channel = self.active_out_channel
            x = self.conv(x)
        return x

    @property
    def module_str(self):
        return "DyShortcut(O%d, R%d)" % (self.active_out_channel, self.reduction)

    @property
    def config(self):
        return {
            "name": DynamicShortcutLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "reduction": self.reduction,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicShortcutLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ShortcutLayer(in_channel, self.active_out_channel, self.reduction)
        sub_layer = sub_layer.to(self.parameters().__next__().device)

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :]
        )

        return sub_layer


# ---- bignas/cnn.py (verbatim) ----


class BigNASStaticModel(nn.Module):
    def __init__(self, first_conv, blocks, last_conv, classifier, resolution, use_v3_head=True):
        super(BigNASStaticModel, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = last_conv
        self.classifier = classifier

        self.resolution = resolution  # input size
        self.use_v3_head = use_v3_head

    def forward(self, x):
        # resize input to target resolution first
        # Rule: transform images into different sizes
        if x.size(-1) != self.resolution:
            x = F.interpolate(x, size=self.resolution, mode="bicubic")

        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        if not self.use_v3_head:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            "name": BigNASStaticModel.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "classifier": self.classifier.config,
            "resolution": self.resolution,
        }

    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                }
        return None

    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                m.training = True
                m.momentum = None  # cumulative moving average
                m.reset_running_stats()


class BigNASDynamicModel(nn.Module):
    def __init__(self, supernet_cfg, n_classes=1000, bn_param=(0.0, 1e-5)):
        super(BigNASDynamicModel, self).__init__()

        self.supernet_cfg = supernet_cfg
        self.n_classes = n_classes
        self.use_v3_head = getattr(self.supernet_cfg, "use_v3_head", False)
        self.stage_names = [
            "first_conv",
            "mb1",
            "mb2",
            "mb3",
            "mb4",
            "mb5",
            "mb6",
            "mb7",
            "last_conv",
        ]

        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list = [], [], [], []
        for name in self.stage_names:
            block_cfg = getattr(self.supernet_cfg, name)
            self.width_list.append(block_cfg.c)
            if name.startswith("mb"):
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
        self.resolution_list = self.supernet_cfg.resolutions

        self.cfg_candidates = {
            "resolution": self.resolution_list,
            "width": self.width_list,
            "depth": self.depth_list,
            "kernel_size": self.ks_list,
            "expand_ratio": self.expand_ratio_list,
        }

        # first conv layer, including conv, bn, act
        out_channel_list, act_func, stride = (
            self.supernet_cfg.first_conv.c,
            self.supernet_cfg.first_conv.act_func,
            self.supernet_cfg.first_conv.s,
        )
        self.first_conv = DynamicConvLayer(
            in_channel_list=val2list(3),
            out_channel_list=out_channel_list,
            kernel_size=3,
            stride=stride,
            act_func=act_func,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names[1:-1]):
            block_cfg = getattr(self.supernet_cfg, key)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else 1
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = (
                        [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                    )
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim,
                    out_channel_list=output_channel,
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet_cfg, "channels_per_group", 1),
                )
                # Rule: add skip-connect, and use 2x2 AvgPool or 1x1 Conv for adaptation
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=stride)
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        last_channel, act_func = self.supernet_cfg.last_conv.c, self.supernet_cfg.last_conv.act_func
        if not self.use_v3_head:
            self.last_conv = DynamicConvLayer(
                in_channel_list=feature_dim,
                out_channel_list=last_channel,
                kernel_size=1,
                act_func=act_func,
            )
        else:
            expand_feature_dim = [f_dim * 6 for f_dim in feature_dim]
            self.last_conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "final_expand_layer",
                            DynamicConvLayer(
                                feature_dim,
                                expand_feature_dim,
                                kernel_size=1,
                                use_bn=True,
                                act_func=act_func,
                            ),
                        ),
                        ("pool", nn.AdaptiveAvgPool2d((1, 1))),
                        (
                            "feature_mix_layer",
                            DynamicConvLayer(
                                in_channel_list=expand_feature_dim,
                                out_channel_list=last_channel,
                                kernel_size=1,
                                act_func=act_func,
                                use_bn=False,
                            ),
                        ),
                    ]
                )
            )

        # final conv layer
        self.classifier = DynamicLinearLayer(
            in_features_list=last_channel, out_features=n_classes, bias=True
        )

        self.init_model()

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        self.active_resolution = 224

    # Rule: Initialize learnable coefficient \gamma=0
    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    if (
                        isinstance(m.mobile_inverted_conv, DynamicMBConvLayer)
                        and m.shortcut is not None
                    ):
                        m.mobile_inverted_conv.point_linear.bn.bn.weight.zero_()

    def init_model(self, model_init="he_fout"):
        """Conv2d, BatchNorm2d, BatchNorm1d, Linear,"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == "he_fout":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif model_init == "he_fin":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def name():
        return "BigNASDynamicModel"

    def forward(self, x):
        # resize input to target resolution first
        if x.size(-1) != self.active_resolution:
            x = F.interpolate(x, size=self.active_resolution, mode="bicubic")

        # first conv
        x = self.first_conv(x)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.last_conv(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = torch.squeeze(x)

        if self.active_dropout_rate > 0 and self.training:
            x = F.dropout(x, p=self.active_dropout_rate)

        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        if not self.use_v3_head:
            _str += self.last_conv.module_str + "\n"
        else:
            _str += self.last_conv.final_expand_layer.module_str + "\n"
            _str += self.last_conv.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": BigNASDynamicModel.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "last_conv": self.last_conv.config if not self.use_v3_head else None,
            "final_expand_layer": self.last_conv.final_expand_layer if self.use_v3_head else None,
            "feature_mix_layer": self.last_conv.feature_mix_layer if self.use_v3_head else None,
            "classifier": self.classifier.config,
            "resolution": self.active_resolution,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                }
        return None

    """ set, sample and get active sub-networks """

    def set_active_subnet(
        self, resolution=224, width=None, depth=None, kernel_size=None, expand_ratio=None, **kwargs
    ):
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width) - 2
        # set resolution
        self.active_resolution = resolution

        # first conv
        self.first_conv.active_out_channel = width[0]

        for stage_id, (c, k, e, d) in enumerate(zip(width[1:-1], kernel_size, expand_ratio, depth)):
            start_idx, end_idx = (
                min(self.block_group_info[stage_id]),
                max(self.block_group_info[stage_id]),
            )  # noqa: F841
            for block_id in range(start_idx, start_idx + d):
                block = self.blocks[block_id]
                # block output channels
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c

                # dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k

                # dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e

        # IRBlocks repated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        # last conv
        if not self.use_v3_head:
            self.last_conv.active_out_channel = width[-1]
        else:
            # default expansion ratio: 6
            self.last_conv.final_expand_layer.active_out_channel = width[-2] * 6
            self.last_conv.feature_mix_layer.active_out_channel = width[-1]

    def get_active_subnet_settings(self):
        r = self.active_resolution
        width, depth, kernel_size, expand_ratio = [], [], [], []

        # first conv
        width.append(self.first_conv.active_out_channel)
        for stage_id in range(len(self.block_group_info)):
            start_idx = min(self.block_group_info[stage_id])
            block = self.blocks[start_idx]  # first block
            width.append(block.mobile_inverted_conv.active_out_channel)
            kernel_size.append(block.mobile_inverted_conv.active_kernel_size)
            expand_ratio.append(block.mobile_inverted_conv.active_expand_ratio)
            depth.append(self.runtime_depth[stage_id])

        if not self.use_v3_head:
            width.append(self.last_conv.active_out_channel)
        else:
            width.append(self.last_conv.feature_mix_layer.active_out_channel)

        return {
            "resolution": r,
            "width": width,
            "kernel_size": kernel_size,
            "expand_ratio": expand_ratio,
            "depth": depth,
        }

    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout
        for idx, block in enumerate(self.blocks):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            this_drop_connect_rate = drop_connect * float(idx) / len(self.blocks)
            block.drop_connect_rate = this_drop_connect_rate

    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net=True)

    def sample_max_subnet(self):
        return self._sample_active_subnet(max_net=True)

    def sample_active_subnet(self, compute_flops=False):
        cfg = self._sample_active_subnet(False, False)
        if compute_flops:
            cfg["flops"] = self.compute_active_subnet_flops()
        return cfg

    def sample_active_subnet_within_range(self, targeted_min_flops, targeted_max_flops):
        while True:
            cfg = self._sample_active_subnet()
            cfg["flops"] = self.compute_active_subnet_flops()
            if cfg["flops"] >= targeted_min_flops and cfg["flops"] <= targeted_max_flops:
                return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False):
        sample_cfg = lambda candidates, sample_min, sample_max: (
            min(candidates)
            if sample_min
            else (max(candidates) if sample_max else random.choice(candidates))
        )  # noqa: E731

        cfg = {}
        # sample a resolution
        cfg["resolution"] = sample_cfg(self.cfg_candidates["resolution"], min_net, max_net)
        for k in ["width", "depth", "kernel_size", "expand_ratio"]:
            cfg[k] = []
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(val2list(vv), min_net, max_net))

        self.set_active_subnet(
            cfg["resolution"], cfg["width"], cfg["depth"], cfg["kernel_size"], cfg["expand_ratio"]
        )
        return cfg

    def mutate_and_reset(self, cfg, prob=0.1, keep_resolution=False):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: (
            x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        )  # noqa: E731
        # sample a resolution
        r = random.random()
        if r < prob and not keep_resolution:
            cfg["resolution"] = pick_another(cfg["resolution"], self.cfg_candidates["resolution"])

        # sample channels, depth, kernel_size, expand_ratio
        for k in ["width", "depth", "kernel_size", "expand_ratio"]:
            for _i, _v in enumerate(cfg[k]):
                r = random.random()
                if r < prob:
                    cfg[k][_i] = pick_another(cfg[k][_i], val2list(self.cfg_candidates[k][_i]))

        self.set_active_subnet(
            cfg["resolution"], cfg["width"], cfg["depth"], cfg["kernel_size"], cfg["expand_ratio"]
        )
        return cfg

    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}
        cfg["resolution"] = cfg1["resolution"] if random.random() < p else cfg2["resolution"]
        for k in ["width", "depth", "kernel_size", "expand_ratio"]:
            cfg[k] = _cross_helper(cfg1[k], cfg2[k], p)

        self.set_active_subnet(
            cfg["resolution"], cfg["width"], cfg["depth"], cfg["kernel_size"], cfg["expand_ratio"]
        )
        return cfg

    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight)

            blocks = []
            input_channel = first_conv.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(
                        ResidualBlock(
                            self.blocks[idx].mobile_inverted_conv.get_active_subnet(
                                input_channel, preserve_weight
                            ),
                            self.blocks[idx].shortcut.get_active_subnet(
                                input_channel, preserve_weight
                            )
                            if self.blocks[idx].shortcut is not None
                            else None,
                        )
                    )
                    input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks += stage_blocks

            if not self.use_v3_head:
                last_conv = self.last_conv.get_active_subnet(input_channel, preserve_weight)
                in_features = last_conv.out_channels
            else:
                final_expand_layer = self.last_conv.final_expand_layer.get_active_subnet(
                    input_channel, preserve_weight
                )
                feature_mix_layer = self.last_conv.feature_mix_layer.get_active_subnet(
                    input_channel * 6, preserve_weight
                )
                in_features = feature_mix_layer.out_channels
                last_conv = nn.Sequential(
                    final_expand_layer, nn.AdaptiveAvgPool2d((1, 1)), feature_mix_layer
                )

            classifier = self.classifier.get_active_subnet(in_features, preserve_weight)

            _subnet = BigNASStaticModel(
                first_conv,
                blocks,
                last_conv,
                classifier,
                self.active_resolution,
                use_v3_head=self.use_v3_head,
            )
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet

    def compute_active_subnet_flops(self):
        def count_conv(c_in, c_out, size_out, groups, k):
            kernel_ops = k**2
            output_elements = c_out * size_out**2
            ops = c_in * output_elements * kernel_ops / groups
            return ops

        def count_linear(c_in, c_out):
            return c_in * c_out

        total_ops = 0

        c_in = 3
        size_out = self.active_resolution // self.first_conv.stride
        c_out = self.first_conv.active_out_channel

        total_ops += count_conv(c_in, c_out, size_out, 1, 3)
        c_in = c_out

        # mb blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                block = self.blocks[idx]
                c_middle = make_divisible(
                    round(c_in * block.mobile_inverted_conv.active_expand_ratio), 8
                )
                # 1*1 conv
                if block.mobile_inverted_conv.inverted_bottleneck is not None:
                    total_ops += count_conv(c_in, c_middle, size_out, 1, 1)
                # dw conv
                stride = 1 if idx > active_idx[0] else block.mobile_inverted_conv.stride
                if size_out % stride == 0:
                    size_out = size_out // stride
                else:
                    size_out = (size_out + 1) // stride
                total_ops += count_conv(
                    c_middle,
                    c_middle,
                    size_out,
                    c_middle,
                    block.mobile_inverted_conv.active_kernel_size,
                )
                # 1*1 conv
                c_out = block.mobile_inverted_conv.active_out_channel
                total_ops += count_conv(c_middle, c_out, size_out, 1, 1)
                # se
                if block.mobile_inverted_conv.use_se:
                    num_mid = make_divisible(
                        c_middle // block.mobile_inverted_conv.depth_conv.se.reduction, divisor=8
                    )
                    total_ops += count_conv(c_middle, num_mid, 1, 1, 1) * 2
                if block.shortcut and c_in != c_out:
                    total_ops += count_conv(c_in, c_out, size_out, 1, 1)
                c_in = c_out

        if not self.use_v3_head:
            c_out = self.last_conv.active_out_channel
            total_ops += count_conv(c_in, c_out, size_out, 1, 1)
        else:
            c_expand = self.last_conv.final_expand_layer.active_out_channel
            c_out = self.last_conv.feature_mix_layer.active_out_channel
            total_ops += count_conv(c_in, c_expand, size_out, 1, 1)
            total_ops += count_conv(c_expand, c_out, 1, 1, 1)

        # n_classes
        total_ops += count_linear(c_out, self.n_classes)
        return total_ops / 1e6

    def load_weights_from_pretrained_models(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint["model_state"]
        for k, v in self.state_dict().items():
            name = k
            v.copy_(pretrained_state_dicts[name])


# ---- menagerie staging glue (not part of the vendored repo) ----
#
# The repo's `_BigNAS_CNN()` factory (configs/search.yaml + configs/train.yaml)
# reads a yacs CfgNode built from a full training config file. We reproduce a tiny,
# self-contained slice of the repo's own `SEARCH_CFG_SETS` (configs/search.yaml)
# as a plain-attribute namespace (the repo's `supernet_cfg.first_conv.c` etc. access
# pattern only needs attribute access, not yacs itself) so `BigNASDynamicModel` builds
# exactly as the repo intends, then extract one concrete sampled subnet via the real
# `get_active_subnet()` entry point -- this is the model the repo evaluates/deploys.
class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _tiny_supernet_cfg():
    # Mirrors configs/search.yaml's SEARCH_CFG_SETS (single-choice = "no search" slice
    # of the full SUPERNET_CFG search space) at greatly reduced channel widths so the
    # traced model is small.
    return _AttrDict(
        use_v3_head=False,
        resolutions=[32],
        channels_per_group=1,
        first_conv=_AttrDict(c=[8], act_func="relu6", s=2),
        mb1=_AttrDict(c=[8], d=[1], k=[3], t=[1], s=1, act_func="relu6", se=False),
        mb2=_AttrDict(c=[12], d=[1], k=[3], t=[4], s=2, act_func="relu6", se=False),
        mb3=_AttrDict(c=[16], d=[1], k=[3], t=[4], s=2, act_func="relu6", se=True),
        mb4=_AttrDict(c=[24], d=[1], k=[3], t=[4], s=2, act_func="relu6", se=False),
        mb5=_AttrDict(c=[32], d=[1], k=[3], t=[4], s=1, act_func="relu6", se=True),
        mb6=_AttrDict(c=[48], d=[1], k=[3], t=[4], s=1, act_func="relu6", se=False),
        mb7=_AttrDict(c=[64], d=[1], k=[3], t=[4], s=1, act_func="relu6", se=False),
        last_conv=_AttrDict(c=[128], act_func="relu6"),
    )


def build_bignas():
    supernet = BigNASDynamicModel(_tiny_supernet_cfg(), n_classes=10, bn_param=(0.1, 1e-5))
    supernet.train()  # DynamicBatchNorm2d.forward() requires training=True
    supernet.set_active_subnet(
        resolution=32,
        width=[8, 8, 12, 16, 24, 32, 48, 64, 128],
        depth=[1, 1, 1, 1, 1, 1, 1],
        kernel_size=[3, 3, 3, 3, 3, 3, 3],
        expand_ratio=[1, 4, 4, 4, 4, 4, 4],
    )
    model = supernet.get_active_subnet(preserve_weight=True)
    model.set_bn_param(momentum=0.1, eps=1e-5)
    model.eval()
    return model


def example_input_bignas():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("BigNAS", "build_bignas", "example_input_bignas", 2020, "vendored"),
]
