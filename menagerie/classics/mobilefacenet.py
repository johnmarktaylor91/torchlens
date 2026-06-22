"""MobileFaceNet -- Efficient CNNs for Face Verification on Mobile Devices.

Chen et al., CCBR 2018.
Paper: https://arxiv.org/abs/1804.07573
Source: https://github.com/Softmax1993/MobileFaceNet

MobileFaceNet combines MobileNetV2 inverted-residual blocks with a global
depthwise convolution (GDConv) that collapses spatial dimensions to a single
feature vector. PReLU activations replace ReLU throughout. This design is
specifically tuned for face embedding with minimal compute.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted-residual block with PReLU."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand: int = 6) -> None:
        super().__init__()
        mid_c = in_c * expand
        self.use_skip = in_c == out_c and stride == 1
        self.conv = nn.Sequential(
            # Pointwise expand
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
            # Depthwise
            nn.Conv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(mid_c),
            # Pointwise project
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_skip:
            return x + out
        return out


class GDConv(nn.Module):
    """Global depthwise conv: kernel equals the spatial size of the input."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, 1, 0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.dw(x))


class MobileFaceNet(nn.Module):
    """Compact MobileFaceNet for 64x64 face input -> 128-dim embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        # Depthwise conv after stem
        self.dw_stem = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.b1 = InvertedResidual(64, 64, stride=2, expand=2)  # 16x16
        self.b2 = InvertedResidual(64, 64, stride=1, expand=2)
        self.b3 = InvertedResidual(64, 128, stride=2, expand=4)  # 8x8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )
        # GDConv collapses 8x8 spatial to 1x1
        self.gdconv = GDConv(512, kernel=8)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # 32x32
        x = self.dw_stem(x)
        x = self.b1(x)  # 16x16
        x = self.b2(x)
        x = self.b3(x)  # 8x8
        x = self.bottleneck(x)
        x = self.gdconv(x)  # 1x1
        x = self.flatten(x)
        x = self.fc(x)
        return x


def build_mobilefacenet() -> nn.Module:
    """Build compact MobileFaceNet (output: (1, 128))."""
    return MobileFaceNet()


def example_input() -> torch.Tensor:
    """Example face image tensor ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "MobileFaceNet (inverted-residual + GDConv face embedding)",
        "build_mobilefacenet",
        "example_input",
        "2018",
        "DC",
    ),
]
