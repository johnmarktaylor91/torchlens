"""MODNet -- Is a Green Screen Really Necessary for Real-Time Portrait Matting?

Ke et al., AAAI 2022.
Paper: https://arxiv.org/abs/2011.11961
Source: https://github.com/ZHKKKe/MODNet

MODNet is a trimap-free portrait matting network with three branches:
(1) Semantic branch: extracts global context via a lightweight encoder + global
    average pool + FC layer, providing coarse alpha at full resolution.
(2) Detail branch: refines edges from early low-stride features.
(3) Fusion branch: combines semantic and detail predictions.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvRes(nn.Module):
    """MobileNetV2-style inverted-residual block with ReLU6."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand: int = 6) -> None:
        super().__init__()
        mid_c = in_c * expand
        self.use_skip = in_c == out_c and stride == 1
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU6(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU6(inplace=True),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw2(self.dw(self.pw1(x)))
        if self.use_skip:
            return x + out
        return out


class MODNet(nn.Module):
    """Compact MODNet for 64x64 portrait input -> alpha matte (1, 1, 64, 64)."""

    def __init__(self) -> None:
        super().__init__()
        # Shared encoder
        self.enc1 = InvRes(3, 16, stride=2, expand=1)  # 32x32
        self.enc2 = InvRes(16, 32, stride=2, expand=6)  # 16x16
        self.enc3 = InvRes(32, 64, stride=2, expand=6)  # 8x8

        # Semantic branch: global context -> coarse alpha
        self.sem_pool = nn.AdaptiveAvgPool2d(1)
        self.sem_fc = nn.Linear(64, 1)

        # Detail branch: from 32x32 features -> alpha
        self.detail_conv = nn.Conv2d(16, 1, 3, padding=1)

        # Fusion branch: combine semantic + detail
        self.fuse = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Shared encoder
        f1 = self.enc1(x)  # (B, 16, 32, 32)
        f2 = self.enc2(f1)  # (B, 32, 16, 16)
        f3 = self.enc3(f2)  # (B, 64, 8, 8)

        # Semantic branch
        sem_feat = self.sem_pool(f3)  # (B, 64, 1, 1)
        sem = self.sem_fc(sem_feat.flatten(1)).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        sem = torch.sigmoid(sem)
        sem = F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)

        # Detail branch
        detail = self.detail_conv(f1)  # (B, 1, 32, 32)
        detail = F.interpolate(detail, size=(H, W), mode="bilinear", align_corners=False)

        # Fusion
        alpha = self.fuse(torch.cat([sem, detail], dim=1))
        return alpha


def build_modnet_portrait_matting() -> nn.Module:
    """Build compact MODNet portrait matting network."""
    return MODNet()


def example_input() -> torch.Tensor:
    """Example RGB portrait tensor ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "MODNet (trimap-free portrait matting)",
        "build_modnet_portrait_matting",
        "example_input",
        "2020",
        "DC",
    ),
]
