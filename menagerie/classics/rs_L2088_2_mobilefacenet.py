# SOURCE: vendored from Xiaoccer/MobileFaceNet_Pytorch @ 26bd8d262017e472a3c7f56d62929e2138a70aeb
# https://raw.githubusercontent.com/Xiaoccer/MobileFaceNet_Pytorch/master/core/model.py
#
# MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices.
# Chen, Liu, Gao, Han. CCBR 2018 (arXiv:1804.07573). MobileNetV2-style inverted-residual
# ("Bottleneck") blocks with PReLU, a depthwise 7x6 "linear" GDConv head (matches the
# 112x96 LFW-aligned-face input), and a final linear embedding projection to 128-d.
#
# `Bottleneck` / `ConvBlock` / `MobileFacenet` are reproduced verbatim from the real
# `core/model.py` (bottleneck_setting tables, forward wiring, and the real Kaiming-style
# conv init loop all unmodified). `ArcMarginProduct` (the real repo's ArcFace training head)
# is intentionally NOT vendored here: it hardcodes `device='cuda'` in its forward pass and
# requires ground-truth `label` input, making it a training-time loss-head, not part of the
# network's inference-time architecture -- MobileFaceNet is used/cited as the `MobileFacenet`
# backbone (128-d face embedding) alone, exactly as e.g. InsightFace deployments consume it.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False
            ),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1],
]


class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x


def build_mobilefacenet():
    torch.manual_seed(0)
    model = MobileFacenet()
    model.eval()
    return model


def example_input_mobilefacenet():
    torch.manual_seed(0)
    return torch.randn(1, 3, 112, 96)


MENAGERIE_ENTRIES = [
    ("MobileFaceNet", "build_mobilefacenet", "example_input_mobilefacenet", 2018, MENAGERIE_ZOO),
]
