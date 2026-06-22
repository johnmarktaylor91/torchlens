"""IResNet -- Improved Residual Networks for Face Recognition.

Deng et al., CVPR 2019.
Paper: https://arxiv.org/abs/1801.07698
Source: https://github.com/deepinsight/insightface

IResNet uses a BN-first residual block: BN -> Conv -> BN -> PReLU -> Conv -> BN,
with a skip projection when channel counts or stride change. This ordering
improves training stability and performance on face recognition tasks.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IResBlock(nn.Module):
    """BN-first residual block used in IResNet."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        if in_c != out_c or stride != 1:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out + self.shortcut(x)


class IResNet(nn.Module):
    """Compact IResNet face backbone for 64x64 input -> 512-dim embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2, stride=2),  # 32x32
        )
        self.layer1 = IResBlock(64, 64)
        self.layer2 = IResBlock(64, 128, stride=2)  # 16x16
        self.layer3 = IResBlock(128, 256, stride=2)  # 8x8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn_out = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.bn_out(x)
        x = self.fc(x)
        return x


def build_iresnet() -> nn.Module:
    """Build compact IResNet face backbone (output: (1, 512))."""
    return IResNet()


def example_input() -> torch.Tensor:
    """Example face image tensor ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "IResNet (BN-first residual face backbone)",
        "build_iresnet",
        "example_input",
        "2019",
        "DC",
    ),
]
