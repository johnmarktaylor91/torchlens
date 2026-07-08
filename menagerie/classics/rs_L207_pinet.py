# SOURCE: vendored from koyeongmin/PINet_new @ master
#
# https://github.com/koyeongmin/PINet_new
# https://raw.githubusercontent.com/koyeongmin/PINet_new/master/TuSimple/hourglass_network.py
# https://raw.githubusercontent.com/koyeongmin/PINet_new/master/TuSimple/util_hourglass.py
# https://raw.githubusercontent.com/koyeongmin/PINet_new/master/TuSimple/parameters.py
#
# PINet (Ko et al. 2020, IEEE T-ITS / arXiv:2002.06604, "Key Points
# Estimation and Point Instance Segmentation Approach for Lane Detection").
# This vendors the real `lane_detection_network` class from
# `TuSimple/hourglass_network.py` and the real `resize_layer`,
# `hourglass_block`, `hourglass_same`, `Output`, `bottleneck`,
# `bottleneck_down`, `bottleneck_up`, `bottleneck_dilation`, and
# `Conv2D_BatchNorm_Relu` classes from `TuSimple/util_hourglass.py` --
# copied verbatim (only whitespace-preserving copy, no architecture
# changes). Stacked-hourglass keypoint network with per-stage confidence /
# offset / instance-embedding heads (`Output` x3 per `hourglass_block`),
# 4 stacked hourglass stages, intermediate supervision feeding forward via
# `re1`/`re2`/`re3` skip-recombination -- exactly as in the real
# `hourglass_block.forward`.
#
# `util_hourglass.py` reads a single global constant from the repo's
# `Parameters` class: `p.feature_size` (instance-embedding channel count,
# used by `hourglass_block.__init__` to size `self.out_instance`). Rather
# than vendor the entire `Parameters` class (which also carries training
# hyperparameters, loss weights, and augmentation ratios unrelated to the
# architecture), this file defines the single real value directly
# (`feature_size = 4`, taken verbatim from `TuSimple/parameters.py`).
#
# Real input is a `(N, 3, 256, 512)` RGB image tensor (`Parameters.x_size
# = 512`, `Parameters.y_size = 256`); `resize_layer` downsamples by 8x
# (stride-2 x3) before the hourglass stack, matching
# `Parameters.resize_ratio = 8`.

import torch
from torch import nn

# ---------------------------------------------------------------------------
# TuSimple/util_hourglass.py (real PINet building blocks)
# ---------------------------------------------------------------------------

# real value from TuSimple/parameters.py: Parameters.feature_size
_FEATURE_SIZE = 4


class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1
    ):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    n_filters,
                    k_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(n_filters),
                nn.PReLU(),
            )
        else:
            self.cbr_unit = nn.Conv2d(
                in_channels,
                n_filters,
                k_size,
                padding=padding,
                stride=stride,
                bias=bias,
                dilation=dilation,
            )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True):
        super(bottleneck, self).__init__()
        self.acti = acti
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1, acti=self.acti)

        self.residual = Conv2D_BatchNorm_Relu(in_channels, out_channels, 1, 0, 1)

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if not self.acti:
            return out

        re = self.residual(x)
        out = out + re

        return out


class bottleneck_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_down, self).__init__()
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 3, 1, 2)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1, dilation=1)
        self.conv3 = nn.Conv2d(temp_channels, out_channels, 1, padding=0, stride=1, bias=True)

        self.residual = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, x, residual=False):
        re = x  # noqa: F841 (kept for fidelity with upstream, unused there too)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if residual:
            return out
        else:
            out = self.prelu(out)

        return out


class bottleneck_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_up, self).__init__()
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, temp_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(temp_channels),
            nn.PReLU(),
        )
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1, dilation=1)

        self.conv3 = nn.Conv2d(temp_channels, out_channels, 1, padding=0, stride=1, bias=True)

        self.residual = nn.Upsample(size=None, scale_factor=2, mode="bilinear")
        self.dropout = nn.Dropout2d(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        re = x  # noqa: F841 (kept for fidelity with upstream, unused there too)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


class bottleneck_dilation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_dilation, self).__init__()
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1, dilation=1)
        self.conv3 = nn.Conv2d(temp_channels, out_channels, 1, padding=0, stride=1, bias=True)

        self.dropout = nn.Dropout2d(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, x, residual=False):
        re = x  # noqa: F841 (kept for fidelity with upstream, unused there too)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if residual:
            return out
        else:
            out = self.prelu(out)

        return out


class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size // 2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size // 2, in_size // 4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size // 4, out_size, 1, 0, 1, acti=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class hourglass_same(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hourglass_same, self).__init__()
        self.down1 = bottleneck_down(in_channels, out_channels)
        self.down2 = bottleneck_down(out_channels, out_channels)
        self.down3 = bottleneck_down(out_channels, out_channels)
        self.down4 = bottleneck_down(out_channels, out_channels)

        self.same1 = bottleneck_dilation(out_channels, out_channels)
        self.same2 = bottleneck_dilation(out_channels, out_channels)
        self.same3 = bottleneck_dilation(out_channels, out_channels)
        self.same4 = bottleneck_dilation(out_channels, out_channels)

        self.up1 = bottleneck_up(out_channels, out_channels)
        self.up2 = bottleneck_up(out_channels, out_channels)
        self.up3 = bottleneck_up(out_channels, out_channels)
        self.up4 = bottleneck_up(out_channels, out_channels)

        self.residual1 = bottleneck_down(out_channels, out_channels)
        self.residual2 = bottleneck_down(out_channels, out_channels)
        self.residual3 = bottleneck_down(out_channels, out_channels)
        self.residual4 = bottleneck_down(in_channels, out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.prelu = nn.PReLU()

    def forward(self, inputs):
        outputs1 = self.down1(inputs)  # 64*32 -> 32*16
        outputs2 = self.down2(outputs1)  # 32*16 -> 16*8
        outputs3 = self.down3(outputs2)  # 16*8 -> 8*4
        outputs4 = self.down4(outputs3)  # 8*4 -> 4*2

        outputs = self.same1(outputs4)  # 4*2 -> 4*2
        feature = self.same2(outputs, True)  # 4*2 -> 4*2
        outputs = self.same3(self.prelu(self.bn(feature)))  # 4*2 -> 4*2
        outputs = self.same4(outputs, True)  # 4*2 -> 4*2

        outputs = self.up1(self.prelu(self.bn1(outputs + self.residual1(outputs3, True))))
        outputs = self.up2(self.prelu(self.bn2(outputs + self.residual2(outputs2, True))))
        outputs = self.up3(self.prelu(self.bn3(outputs + self.residual3(outputs1, True))))
        outputs = self.up4(self.prelu(self.bn4(outputs + self.residual4(inputs, True))))

        return outputs, feature


class resize_layer(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True):
        super(resize_layer, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(
            in_channels, out_channels // 4, 3, 1, 2, dilation=1, acti=False
        )
        self.conv2 = Conv2D_BatchNorm_Relu(
            out_channels // 4, out_channels // 2, 3, 1, 2, dilation=1, acti=False
        )
        self.conv3 = Conv2D_BatchNorm_Relu(
            out_channels // 2, out_channels // 1, 3, 1, 2, dilation=1, acti=False
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.bn3 = nn.BatchNorm2d(out_channels // 1)
        self.prelu = nn.PReLU()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.prelu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.prelu(outputs)

        outputs = self.conv3(outputs)

        return outputs


class hourglass_block(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True, input_re=True):
        super(hourglass_block, self).__init__()
        self.layer1 = hourglass_same(in_channels, out_channels)
        self.re1 = bottleneck_dilation(out_channels, out_channels)
        self.re2 = nn.Conv2d(
            out_channels, out_channels, 1, padding=0, stride=1, bias=True, dilation=1
        )
        self.re3 = nn.Conv2d(1, out_channels, 1, padding=0, stride=1, bias=True, dilation=1)

        self.out_confidence = Output(out_channels, 1)
        self.out_offset = Output(out_channels, 2)
        self.out_instance = Output(out_channels, _FEATURE_SIZE)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(1)

        self.input_re = input_re

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, inputs):
        inputs_a = self.prelu(self.bn1(inputs))

        outputs, feature = self.layer1(inputs_a)

        outputs_a = self.bn2(outputs)
        outputs_a = self.prelu(outputs_a)
        outputs_a = self.re1(outputs_a)

        outputs = self.re2(outputs_a)

        out_confidence = self.out_confidence(outputs_a)
        out_offset = self.out_offset(outputs_a)
        out_instance = self.out_instance(outputs_a)

        out = self.prelu(self.bn3(out_confidence))
        out = self.re3(out)

        if self.input_re:
            outputs = outputs + out + inputs
        else:
            outputs = outputs + out

        return [out_confidence, out_offset, out_instance], outputs, feature


# ---------------------------------------------------------------------------
# TuSimple/hourglass_network.py (real PINet top-level network)
# ---------------------------------------------------------------------------


class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        self.resizing = resize_layer(3, 128)

        # feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)
        self.layer3 = hourglass_block(128, 128)
        self.layer4 = hourglass_block(128, 128)

    def forward(self, inputs):
        # feature extraction
        out = self.resizing(inputs)
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)
        result3, out, feature3 = self.layer3(out)
        result4, out, feature4 = self.layer4(out)

        return [result1, result2, result3, result4], [feature1, feature2, feature3, feature4]


def build_pinet():
    return lane_detection_network()


def example_input_pinet():
    # real input size: Parameters.x_size=512, Parameters.y_size=256 (TuSimple/parameters.py)
    return torch.randn(1, 3, 256, 512)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PINet", "build_pinet", "example_input_pinet", 2020, "vendored"),
]
