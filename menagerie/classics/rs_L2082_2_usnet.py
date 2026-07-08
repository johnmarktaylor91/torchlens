# SOURCE: vendored from JiahuiYu/slimmable_networks @ master
#
# US-Net: "Universally Slimmable Networks and Improved Training Techniques" (Yu &
# Huang, ICCV 2019). Same repo as Slimmable Networks (Yu et al., ICLR 2019); US-Net
# generalizes slimmable width-switching from a small fixed set of widths to ANY
# width in a continuous range via `USConv2d`/`USBatchNorm2d`/`USLinear` ("universally
# slimmable" ops) plus the sandwich rule + in-place distillation training recipe.
#
# The real model code (models/slimmable_ops.py::USConv2d/USBatchNorm2d/USLinear/
# make_divisible, models/us_mobilenet_v2.py::InvertedResidual/Model) imports ONLY
# torch.nn plus one repo-local module: `from utils.config import FLAGS`, a global
# argparse+yaml config singleton. `FLAGS` is not architecture -- it's the repo's CLI
# config plumbing -- so it is vendored here as a tiny local namespace populated with
# the REAL values from the repo's own released US-MobileNetV2 ImageNet config
# (apps/us_mobilenet_v2_train_val.yml: width_mult_range=[0.35, 1.0], the 27-point
# width_mult_list used for calibrated-BN switches, dataset='imagenet1k',
# reset_parameters=True). No architectural code was rewritten; only the FLAGS import
# was swapped for an equivalent local singleton with identical real field values, and
# relative `from .slimmable_ops import ...` was flattened to same-module references.

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class _Flags:
    """Vendored equivalent of utils.config.FLAGS, populated with the real values
    from apps/us_mobilenet_v2_train_val.yml (US-MobileNetV2 ImageNet1k config)."""

    dataset = "imagenet1k"
    width_mult_range = [0.35, 1.0]
    width_mult_list = [
        0.35,
        0.375,
        0.4,
        0.425,
        0.45,
        0.475,
        0.5,
        0.525,
        0.55,
        0.575,
        0.6,
        0.625,
        0.65,
        0.675,
        0.7,
        0.725,
        0.75,
        0.775,
        0.8,
        0.825,
        0.85,
        0.875,
        0.9,
        0.925,
        0.95,
        0.975,
        1.0,
    ]
    reset_parameters = True
    conv_averaged = False
    cumulative_bn_stats = True


FLAGS = _Flags()


# ---------------------------------------------------------------------------
# models/slimmable_ops.py (verbatim architecture; only the FLAGS import source
# changed, see header)
# ---------------------------------------------------------------------------
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


class USConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        depthwise=False,
        bias=True,
        us=(True, True),
        ratio=(1, 1),
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = (
                make_divisible(self.in_channels_max * self.width_mult / self.ratio[0])
                * self.ratio[0]
            )
        if self.us[1]:
            self.out_channels = (
                make_divisible(self.out_channels_max * self.width_mult / self.ratio[1])
                * self.ratio[1]
            )
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[: self.out_channels, : self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[: self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        if getattr(FLAGS, "conv_averaged", False):
            y = y * (self.in_channels_max / self.in_channels)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=(True, True)):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_features = make_divisible(self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(self.out_features_max * self.width_mult)
        weight = self.weight[: self.out_features, : self.in_features]
        if self.bias is not None:
            bias = self.bias[: self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super().__init__(num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList(
            [
                nn.BatchNorm2d(i, affine=False)
                for i in [
                    make_divisible(self.num_features_max * width_mult / ratio) * ratio
                    for width_mult in FLAGS.width_mult_list
                ]
            ]
        )
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = make_divisible(self.num_features_max * self.width_mult / self.ratio) * self.ratio
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps,
            )
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps,
            )
        return y


# ---------------------------------------------------------------------------
# models/us_mobilenet_v2.py (verbatim architecture; relative slimmable_ops import
# flattened, FLAGS import source changed, see header)
# ---------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                USConv2d(inp, expand_inp, 1, 1, 0, bias=False, ratio=[1, expand_ratio]),
                USBatchNorm2d(expand_inp, ratio=expand_ratio),
                nn.ReLU6(inplace=False),
            ]
        # depthwise + project back
        layers += [
            USConv2d(
                expand_inp,
                expand_inp,
                3,
                stride,
                1,
                groups=expand_inp,
                depthwise=True,
                bias=False,
                ratio=[expand_ratio, expand_ratio],
            ),
            USBatchNorm2d(expand_inp, ratio=expand_ratio),
            nn.ReLU6(inplace=False),
            USConv2d(expand_inp, outp, 1, 1, 0, bias=False, ratio=[expand_ratio, 1]),
            USBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.residual_connection:
            res = self.body(x)
            res = res + x
        else:
            res = self.body(x)
        return res


class USMobileNetV2(nn.Module):
    """US-Net's universally-slimmable MobileNetV2 (models/us_mobilenet_v2.py::Model)."""

    def __init__(self, num_classes: int = 1000, input_size: int = 224) -> None:
        super().__init__()

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
        if FLAGS.dataset == "cifar10":
            self.block_setting[2] = [6, 24, 2, 1]

        features = []

        width_mult = FLAGS.width_mult_range[-1]
        # head
        assert input_size % 32 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        first_stride = 2
        features.append(
            nn.Sequential(
                USConv2d(3, channels, 3, first_stride, 1, bias=False, us=[False, True]),
                USBatchNorm2d(channels),
                nn.ReLU6(inplace=False),
            )
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(channels, outp, s, t))
                else:
                    features.append(InvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        features.append(
            nn.Sequential(
                USConv2d(channels, self.outp, 1, 1, 0, bias=False, us=[True, False]),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=False),
            )
        )
        avg_pool_size = input_size // 32
        features.append(nn.AvgPool2d(avg_pool_size))

        self.features = nn.Sequential(*features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()

        # US-Net ops read a live `width_mult` attribute at forward time (this is how
        # width-switching during search-free slimmable inference works); pin every
        # slimmable op to the maximum trained width so the model is directly runnable.
        self._set_width_mult(FLAGS.width_mult_range[-1])

    def _set_width_mult(self, width_mult: float) -> None:
        for m in self.modules():
            if isinstance(m, (USConv2d, USLinear, USBatchNorm2d)):
                m.width_mult = width_mult

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def build_us_mobilenet_v2() -> nn.Module:
    """Build a tiny random-init US-MobileNetV2 pinned at full (1.0) width."""

    return USMobileNetV2(num_classes=10, input_size=64).eval()


def example_input_us_mobilenet_v2() -> Tensor:
    """Return a small RGB classification input."""

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "USNet-MobileNetV2",
        "build_us_mobilenet_v2",
        "example_input_us_mobilenet_v2",
        "2019",
        "CV",
    )
]
