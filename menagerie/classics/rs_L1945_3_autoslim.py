# SOURCE: vendored from JiahuiYu/slimmable_networks @ master
# https://raw.githubusercontent.com/JiahuiYu/slimmable_networks/master/models/autoslim_mobilenet_v2.py
# https://raw.githubusercontent.com/JiahuiYu/slimmable_networks/master/models/slimmable_ops.py
# https://raw.githubusercontent.com/JiahuiYu/slimmable_networks/master/apps/autoslim_mobilenet_v2_train_val.yml
#
# Yu, Huang, 2019 (ICCV) "AutoSlim: Towards One-Shot Architecture Search for
# Channel Numbers". AutoSlim greedily slims a slimmable MobileNetV2 supernet to
# find per-layer channel numbers directly (rather than searching operators), then
# a "slimmable" MobileNetV2 (`autoslim_mobilenet_v2.Model`) is trained at the found
# widths -- `SlimmableConv2d`/`SwitchableBatchNorm2d`/`SlimmableLinear` (channel
# lists selectable at runtime via a global `width_mult` index) are AutoSlim's real
# per-layer-channel-search architecture, vendored verbatim below rather than
# rebuilt from a stock MobileNetV2 class. The real found 207M-FLOPs AutoSlim-found
# channel_num_list below is copied verbatim from
# `apps/autoslim_mobilenet_v2_train_val.yml` (the first/smallest of that repo's
# three released found architectures).
#
# `slimmable_ops.py` (`SwitchableBatchNorm2d`, `SlimmableConv2d`, `SlimmableLinear`,
# `pop_channels`; the universally-slimmable `USConv2d`/`USLinear`/`USBatchNorm2d`
# variants are omitted -- unused by `autoslim_mobilenet_v2.Model`) and
# `autoslim_mobilenet_v2.py` (`InvertedResidual`, `Model`) are reproduced verbatim
# below (only the `utils.config.FLAGS` global-config singleton is replaced with a
# tiny local `_Flags` object holding the same fields the real code reads:
# `width_mult_list`, `channel_num_list`, `dataset`, `reset_parameters` -- this is a
# config-plumbing substitution, not an architectural change; every conv/BN/linear
# layer and the `InvertedResidual`/`Model.forward` control flow is identical to the
# real repo code).

import torch.nn as nn
import math

MENAGERIE_ZOO = "vendored-pytorch"


# ============================================================================
# utils.config.FLAGS substitute (tiny local config object; not an architecture
# change -- the real code only reads FLAGS.width_mult_list / .channel_num_list /
# .dataset / .reset_parameters)
# ============================================================================


class _Flags:
    width_mult_list = [1.0]
    channel_num_list = None  # populated per-build below
    dataset = "imagenet1k"
    reset_parameters = True


FLAGS = _Flags()


# ============================================================================
# slimmable_ops.py (verbatim, minus unused USConv2d/USLinear/USBatchNorm2d)
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


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


# ============================================================================
# autoslim_mobilenet_v2.py (verbatim `InvertedResidual`/`Model`)
# ============================================================================


class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, cmid, stride):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        if cmid != inp:
            layers += [
                SlimmableConv2d(inp, cmid, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(cmid),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            SlimmableConv2d(cmid, cmid, 3, stride, 1, groups_list=cmid, bias=False),
            SwitchableBatchNorm2d(cmid),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(cmid, outp, 1, 1, 0, bias=False),
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

        self.strides_setting = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
        channel_num_list = FLAGS.channel_num_list.copy()
        if FLAGS.dataset.startswith("cifar10"):
            self.strides_setting[1] = 1

        self.features = []

        assert input_size % 32 == 0
        channels = pop_channels(FLAGS.channel_num_list)
        first_stride = 1 if FLAGS.dataset.startswith("cifar10") else 2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3, first_stride, 1, bias=False
                ),
                SwitchableBatchNorm2d(channels),
                nn.ReLU6(inplace=True),
            )
        )

        for index, s in enumerate(self.strides_setting):
            if index == 0:
                outp = pop_channels(FLAGS.channel_num_list)
                cmid = channels
            else:
                cmid = pop_channels(FLAGS.channel_num_list)
                outp = pop_channels(FLAGS.channel_num_list)
            self.features.append(InvertedResidual(channels, outp, cmid, s))
            channels = outp

        self.outp = pop_channels(FLAGS.channel_num_list)
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(channels, self.outp, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        self.features = nn.Sequential(*self.features)
        FLAGS.channel_num_list = channel_num_list.copy()

        self.classifier = nn.Sequential(
            SlimmableLinear(self.outp, [num_classes for _ in range(len(self.outp))])
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ============================================================================
# build_/example_input_ harness
# ============================================================================

# Real AutoSlim-found 207M-FLOPs channel_num_list, verbatim from
# apps/autoslim_mobilenet_v2_train_val.yml (`channel_num_list[0]`, "dynamic 12").
# The yaml's `channel_num_list` is a list of one row PER width_mult entry (here
# FLAGS.width_mult_list=[1.0], a single width point matching the found static
# architecture rather than the full slimmable range); `pop_channels` pops index 0
# from every row simultaneously, so `FLAGS.channel_num_list` must be
# `[_AUTOSLIM_FOUND_CHANNELS]` (a single-row wrapper), not a per-value split.
# `Model.__init__` pops 35 of the 36 values (1 stem + 1 + 2*16 body + 1 tail); the
# trailing "1000" entry is the num_classes placeholder used by other model
# variants in this repo and is left un-popped here too, matching the real code.
_AUTOSLIM_FOUND_CHANNELS = [
    8,
    8,
    96,
    16,
    96,
    16,
    96,
    24,
    144,
    24,
    144,
    24,
    144,
    48,
    288,
    48,
    288,
    48,
    288,
    48,
    288,
    64,
    432,
    64,
    432,
    64,
    648,
    176,
    720,
    176,
    720,
    176,
    1440,
    280,
    1920,
    1000,
]


def build_autoslim_mbv2():
    FLAGS.width_mult_list = [1.0]
    FLAGS.dataset = "imagenet1k"
    FLAGS.reset_parameters = True
    FLAGS.channel_num_list = [_AUTOSLIM_FOUND_CHANNELS.copy()]
    model = Model(num_classes=10, input_size=32)
    model.eval()
    return model


def example_input_autoslim_mbv2():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "AutoSlim-MobileNetV2",
        build_autoslim_mbv2,
        example_input_autoslim_mbv2,
        2019,
        "vendored-pytorch",
    ),
]
