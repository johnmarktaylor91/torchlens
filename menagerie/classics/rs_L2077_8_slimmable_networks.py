# SOURCE: vendored from JiahuiYu/slimmable_networks @ master
# https://raw.githubusercontent.com/JiahuiYu/slimmable_networks/master/models/slimmable_ops.py
# https://raw.githubusercontent.com/JiahuiYu/slimmable_networks/master/models/s_mobilenet_v2.py
#
# Yu, Yang, Xu, Yang, Huang, 2019 (ICLR 2019) "Slimmable Neural Networks".
# `SlimmableConv2d`/`SwitchableBatchNorm2d` (a single conv/BN weight tensor
# sliced down to one of several pre-registered channel-width configurations,
# selected at runtime via a `width_mult` attribute, with one *independent*
# BatchNorm2d per width to avoid the well-known slimmable-BN train/test
# statistics mismatch) trained jointly with the switchable-width MobileNetV2
# backbone IS the paper's whole architectural contribution, so this is
# vendored (real code), not built from a stock MobileNetV2 class.
#
# `models/slimmable_ops.py::SwitchableBatchNorm2d`/`SlimmableConv2d`/
# `SlimmableLinear` and `models/s_mobilenet_v2.py::InvertedResidual`/`Model`
# are reproduced verbatim. The one non-architectural fix: the real repo reads
# `FLAGS.width_mult_list` (and `Model`'s `FLAGS.reset_parameters`) from a
# global `utils.config.FLAGS` singleton populated by `utils/config.py::app()`,
# which parses a YAML config path off argv/stdin at import time (crashes
# outside the repo's own `train.py` entrypoint) -- `FLAGS` is replaced here
# with a plain namespace holding the same two config keys the vendored code
# actually reads, with the same values the repo's own `configs/*.yml` sample
# configs use (`width_mult_list: [0.35, 0.5, 0.75, 1.0]`,
# `reset_parameters: True`); no code path in `slimmable_ops.py`/
# `s_mobilenet_v2.py` is touched.

import math

import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class _Flags:
    """Minimal stand-in for the real repo's `utils.config.FLAGS` singleton
    (see module header) -- holds only the config keys `slimmable_ops.py`/
    `s_mobilenet_v2.py` actually read."""

    width_mult_list = [0.35, 0.5, 0.75, 1.0]
    reset_parameters = True


FLAGS = _Flags()


# ============================================================================
# models/slimmable_ops.py (verbatim, minus the FLAGS import -- see header)
# ============================================================================


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels_list,
        out_channels_list,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups_list=[1],
        bias=True,
    ):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list),
            max(out_channels_list),
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=max(groups_list),
            bias=bias,
        )
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[: self.out_channels, : self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[: self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias
        )
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[: self.out_features, : self.in_features]
        if self.bias is not None:
            bias = self.bias[: self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


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


# ============================================================================
# models/s_mobilenet_v2.py (verbatim, minus the FLAGS import -- see header)
# ============================================================================


class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = [i * expand_ratio for i in inp]
        if expand_ratio != 1:
            layers += [
                SlimmableConv2d(inp, expand_inp, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SlimmableConv2d(
                expand_inp, expand_inp, 3, stride, 1, groups_list=expand_inp, bias=False
            ),
            SwitchableBatchNorm2d(expand_inp),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(expand_inp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = [make_divisible(32 * width_mult) for width_mult in FLAGS.width_mult_list]
        self.outp = (
            make_divisible(1280 * max(FLAGS.width_mult_list))
            if max(FLAGS.width_mult_list) > 1.0
            else 1280
        )
        first_stride = 2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3, first_stride, 1, bias=False
                ),
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True),
            )
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = [make_divisible(c * width_mult) for width_mult in FLAGS.width_mult_list]
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(channels, outp, s, t))
                else:
                    self.features.append(InvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    channels, [self.outp for _ in range(len(channels))], 1, 1, 0, bias=False
                ),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_slimmable_mobilenet_v2():
    import torch

    torch.manual_seed(0)
    model = Model(num_classes=10, input_size=32)
    model.eval()
    return model


def example_input_slimmable_mobilenet_v2():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Slimmable Networks (S-MobileNetV2)",
        "build_slimmable_mobilenet_v2",
        "example_input_slimmable_mobilenet_v2",
        2019,
        "vendored-pytorch",
    ),
]
