# SOURCE: vendored from dwofk/fast-depth @ master
# https://raw.githubusercontent.com/dwofk/fast-depth/master/models.py
# https://raw.githubusercontent.com/dwofk/fast-depth/master/imagenet/mobilenet.py
#
# Wofk et al. 2019 (ICRA) "FastDepth: Fast Monocular Depth Estimation on Embedded Systems" --
# MobileNet encoder (imagenet/mobilenet.py, copied verbatim) feeding a depthwise-separable
# decoder with additive skip connections (models.py::MobileNetSkipAdd, copied verbatim). This
# is the flagship "mobilenet-nnconv5dw-skipadd" configuration from the paper (178 fps on
# Jetson TX2). Only the `pretrained=True` ImageNet-checkpoint-loading branch is skipped here
# (random init via `pretrained=False`, an existing code path in the original class); the
# `imagenet.mobilenet` import is inlined since this is a single-file staging module.
"""FastDepth: MobileNet encoder + depthwise-separable skip-add decoder for monocular depth."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from imagenet/mobilenet.py ---
class MobileNet(nn.Module):
    def __init__(self, relu6=True):
        super(MobileNet, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2, relu6),
            conv_dw(32, 64, 1, relu6),
            conv_dw(64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6),
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


# --- vendored from models.py ---
def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def depthwise(in_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(
        kernel_size, padding
    )
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            groups=in_channels,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetSkipAdd(nn.Module):
    def __init__(self, output_size, pretrained=False):
        super(MobileNetSkipAdd, self).__init__()
        self.output_size = output_size
        mobilenet = MobileNet()
        if pretrained:
            raise NotImplementedError("ImageNet-pretrained checkpoint loading is not vendored")
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr(self, "conv{}".format(i), mobilenet.model[i])

        kernel_size = 5
        self.decode_conv1 = nn.Sequential(depthwise(1024, kernel_size), pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(depthwise(512, kernel_size), pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(depthwise(256, kernel_size), pointwise(256, 128))
        self.decode_conv4 = nn.Sequential(depthwise(128, kernel_size), pointwise(128, 64))
        self.decode_conv5 = nn.Sequential(depthwise(64, kernel_size), pointwise(64, 32))
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        x1 = x2 = x3 = None
        for i in range(14):
            layer = getattr(self, "conv{}".format(i))
            x = layer(x)
            if i == 1:
                x1 = x
            elif i == 3:
                x2 = x
            elif i == 5:
                x3 = x
        for i in range(1, 6):
            layer = getattr(self, "decode_conv{}".format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i == 4:
                x = x + x1
            elif i == 3:
                x = x + x2
            elif i == 2:
                x = x + x3
        x = self.decode_conv6(x)
        return x


def build_fastdepth():
    return MobileNetSkipAdd(output_size=(224, 224), pretrained=False)


def example_input_fastdepth():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 224, 224),)


MENAGERIE_ENTRIES = [
    ("FastDepth", "build_fastdepth", "example_input_fastdepth", 2019, "vendored"),
]
