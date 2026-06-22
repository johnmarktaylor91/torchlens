"""SqueezeNext: hardware-aware CNN blocks.

Paper: "SqueezeNext: Hardware-Aware Neural Network Design", Gholami et al.,
CVPR Workshops 2018.

The compact model keeps the SqueezeNext block: two-stage channel squeeze,
factorized ``1 x N`` and ``N x 1`` convolutions, final expansion, residual
projection, and a final bottleneck before classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeNextBlock(nn.Module):
    """Two-stage squeeze and separable spatial bottleneck block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize a SqueezeNext residual block."""

        super().__init__()
        mid1 = max(out_ch // 2, 4)
        mid2 = max(out_ch // 4, 4)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid1, 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid2, 1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, mid2, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, mid2, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual SqueezeNext block."""

        return torch.relu(self.body(x) + self.skip(x))


class SqueezeNextCompact(nn.Module):
    """Compact SqueezeNext classifier."""

    def __init__(self) -> None:
        """Initialize SqueezeNext stages."""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.blocks = nn.Sequential(
            SqueezeNextBlock(16, 32),
            SqueezeNextBlock(32, 48, stride=2),
            SqueezeNextBlock(48, 64),
        )
        self.final_bottleneck = nn.Conv2d(64, 32, 1)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image."""

        x = self.blocks(self.stem(x))
        x = torch.relu(self.final_bottleneck(x)).mean(dim=(2, 3))
        return self.head(x)


def build() -> nn.Module:
    """Build compact SqueezeNext."""

    return SqueezeNextCompact()


def example_input() -> torch.Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("SqueezeNext", "build", "example_input", "2018", "E7")]
