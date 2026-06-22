"""GhostFaceNet -- Lightweight Face Recognition with Ghost Modules.

He et al., 2023.
Paper: https://arxiv.org/abs/2301.00535
Source: https://github.com/Hazqeel09/ellzaf_ml

GhostFaceNet applies the GhostNet backbone to face recognition: a stem conv,
GhostBottleneck stages, then a global pool and linear projection. Ghost modules
produce feature maps by pairing a primary conv (half channels) with a cheap
depthwise conv on those features (other half), then concatenating.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GhostModule(nn.Module):
    """Ghost module: primary conv + cheap depthwise conv, concat."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int = 1,
        dw_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        init_c = out_c // 2
        cheap_c = out_c - init_c
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, init_c, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(init_c),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(
                init_c,
                cheap_c,
                dw_size,
                1,
                dw_size // 2,
                groups=init_c,
                bias=False,
            ),
            nn.BatchNorm2d(cheap_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost bottleneck with optional strided depthwise and skip projection."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        mid_c = out_c * 2
        self.ghost1 = GhostModule(in_c, mid_c)
        if stride > 1:
            self.dw: nn.Module = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, kernel, stride, kernel // 2, groups=mid_c, bias=False),
                nn.BatchNorm2d(mid_c),
            )
        else:
            self.dw = nn.Identity()
        self.ghost2 = GhostModule(mid_c, out_c)
        if in_c != out_c or stride != 1:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.ghost2(out)
        return out + shortcut


class GhostFaceNet(nn.Module):
    """Compact GhostFaceNet backbone for face embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.b1 = GhostBottleneck(16, 16, stride=1)
        self.b2 = GhostBottleneck(16, 32, stride=2)
        self.b3 = GhostBottleneck(32, 64, kernel=5, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 512)
        self.bn_out = nn.BatchNorm1d(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.bottleneck(x)
        x = self.pool(x).flatten(1)
        x = self.bn_out(self.fc(x))
        return x


def build_ghostfacenet() -> nn.Module:
    """Build compact GhostFaceNet (output: (1, 512))."""
    return GhostFaceNet()


def example_input() -> torch.Tensor:
    """Example face image tensor ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "GhostFaceNet (GhostNet backbone face recognition)",
        "build_ghostfacenet",
        "example_input",
        "2023",
        "DC",
    ),
]
