# SOURCE: vendored from cleardusk/3DDFA_V2 @ master
#
# 3DDFA_V2: "Towards Fast, Accurate and Stable 3D Dense Face Alignment" (Guo, Zhu,
# Yang, Yang, Lei, Li, ECCV 2020). The official inference pipeline (TDDFA.py) wraps
# face-box cropping, a 3DMM (BFM) parameter decoder, and this MobileNet backbone;
# the actual neural network -- the piece TorchLens captures -- is the backbone in
# models/mobilenet_v1.py, which regresses 62 3DMM parameters (12 pose + 40 shape +
# 10 expression) from a cropped 120x120 face crop. This is the real repo's default
# released architecture (configs/mb1_120x120.yml: arch=mobilenet, widen_factor=1.0).
# It is a standard MobileNetV1 (Howard et al. 2017) depthwise-separable-conv stack
# with num_classes repurposed as the 62-d 3DMM parameter output; imports are ONLY
# `math` and `torch.nn` (base libs), so it is vendored verbatim (unmodified
# architecture, only the module-level docstring/comments trimmed).

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DepthWiseBlock(nn.Module):
    """MobileNetV1 depthwise-separable conv block (models/mobilenet_v1.py)."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1, prelu: bool = False) -> None:
        super().__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(
            inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU() if prelu else nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class TDDFAMobileNet(nn.Module):
    """3DDFA_V2's MobileNetV1 3DMM parameter regressor (models/mobilenet_v1.py::MobileNet).

    Real default config (configs/mb1_120x120.yml): widen_factor=1.0, num_classes=62
    (12 pose + 40 shape + 10 expression 3DMM parameters), input size 120x120.
    """

    def __init__(
        self,
        widen_factor: float = 1.0,
        num_classes: int = 62,
        prelu: bool = False,
        input_channel: int = 3,
    ) -> None:
        super().__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(
            input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        self.relu = nn.PReLU() if prelu else nn.ReLU(inplace=False)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def build_tddfa_mobilenet() -> nn.Module:
    """Build the real default 3DDFA_V2 backbone (widen_factor=1.0, num_classes=62)."""

    return TDDFAMobileNet(widen_factor=1.0, num_classes=62).eval()


def example_input_tddfa_mobilenet() -> Tensor:
    """Return a face crop at the real repo's default input size (120x120)."""

    return torch.randn(1, 3, 120, 120)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "3DDFA_V2-MobileNet",
        "build_tddfa_mobilenet",
        "example_input_tddfa_mobilenet",
        "2020",
        "CV",
    )
]
