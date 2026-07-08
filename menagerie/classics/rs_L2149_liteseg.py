# SOURCE: vendored from tahaemara/LiteSeg @ master
# https://raw.githubusercontent.com/tahaemara/LiteSeg/master/models/liteseg_mobilenet.py
# https://raw.githubusercontent.com/tahaemara/LiteSeg/master/models/aspp.py
# https://raw.githubusercontent.com/tahaemara/LiteSeg/master/models/separableconv.py
# https://raw.githubusercontent.com/tahaemara/LiteSeg/master/models/backbone_networks/MobileNetV2.py
#
# Emara, El Sayed, Abd El Munim 2019 "LiteSeg: A Novel Lightweight ConvNet for Semantic
# Segmentation" (DICTA 2019, arXiv:1912.06683) -- deeper dilated ASPP (rates 1/3/6/9) with
# depthwise-separable convs, fused with a MobileNetV2 backbone and low-level-feature skip
# connection for real-time semantic segmentation (default/best-performing backbone variant
# per the paper; the repo also ships ShuffleNet/DarkNet19 backbone variants). RT (LiteSeg
# with MobileNetV2 backbone), ASPP, SeparableConv2d, and MobileNetV2/InvertedResidual are
# copied verbatim from the four source files above; only the `from models... import`
# statements are inlined since this is a single-file staging module, and the pretrained-
# weight-loading branch (`if pretrained: state_dict = torch.load(...)`) is dropped for a
# random-init tiny build (`pretrained=False`).
"""LiteSeg-MobileNet: deeper dilated ASPP + depthwise-separable convs over a MobileNetV2
backbone, for real-time semantic segmentation (Emara et al. 2019)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from models/separableconv.py ---
class SeparableConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
    ):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


# --- vendored from models/aspp.py ---
class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate = rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            self.conv1 = SeparableConv2d(planes, planes, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()

        self.atrous_convolution = SeparableConv2d(inplanes, planes, kernel_size, 1, padding, rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        if self.rate != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# --- vendored from models/backbone_networks/MobileNetV2.py ---
# (original attribution in the source file: "https://github.com/ericsun99/MobileNet-V2-Pytorch,
# edited by: Taha Emara")
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        i = 0
        for i, l in enumerate(self.features[:4]):  # noqa: E741
            x = l(x)
        keep = x
        for z, l in enumerate(self.features[4:]):  # noqa: E741
            x = l(x)
        return x, keep

    def _initialize_weights(self):
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


# --- vendored from models/liteseg_mobilenet.py ---
class RT(nn.Module):
    def __init__(self, n_classes=19, PRETRAINED_WEIGHTS=".", pretrained=True):
        super(RT, self).__init__()

        self.mobile_features = MobileNetV2()
        if pretrained:
            state_dict = torch.load(PRETRAINED_WEIGHTS)
            self.mobile_features.load_state_dict(state_dict)

        rates = [1, 3, 6, 9]

        self.aspp1 = ASPP(1280, 96, rate=rates[0])
        self.aspp2 = ASPP(1280, 96, rate=rates[1])
        self.aspp3 = ASPP(1280, 96, rate=rates[2])
        self.aspp4 = ASPP(1280, 96, rate=rates[3])

        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1280, 96, 1, stride=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.conv1 = SeparableConv2d(480 + 1280, 96, 1)
        self.bn1 = nn.BatchNorm2d(96)

        self.last_conv = nn.Sequential(
            SeparableConv2d(24 + 96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            SeparableConv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, n_classes, kernel_size=1, stride=1),
        )

    def forward(self, input):
        x, low_level_features = self.mobile_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x, x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(
            x,
            size=(int(math.ceil(input.size()[-2] / 4)), int(math.ceil(input.size()[-1] / 4))),
            mode="bilinear",
            align_corners=True,
        )

        x = torch.cat((x, low_level_features), dim=1)

        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def build_liteseg():
    model = RT(n_classes=19, pretrained=False)
    # eval mode: at 128x128 input the ASPP branch's global-average-pool path collapses
    # to a 1x1 spatial map, which BatchNorm2d rejects in training mode for batch size 1
    # ("more than 1 value per channel"); eval mode uses running stats instead.
    model.eval()
    return model


def example_input_liteseg():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 128, 128),)


MENAGERIE_ENTRIES = [
    ("LiteSeg-MobileNet", "build_liteseg", "example_input_liteseg", 2019, "vendored"),
]
