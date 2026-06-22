"""Stacked Hourglass pose network, 2016.

Paper: Stacked Hourglass Networks for Human Pose Estimation (Newell, Yang,
Deng; ECCV 2016).

Faithful compact random-init reconstruction of the distinctive topology:
repeated bottom-up/top-down hourglass modules with residual blocks, skip fusion
at each scale, intermediate heatmap supervision, and feature/logit reinjection
between stacks. The target name is 8-stack, so the compact reconstruction uses
eight shallow hourglass stacks with narrow channels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Bottleneck residual block from the hourglass family."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Create a residual block.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        """
        super().__init__()
        mid = max(out_ch // 2, 1)
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, out_ch, 1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual processing.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """
        return self.skip(x) + self.body(x)


class Hourglass(nn.Module):
    """Recursive bottom-up/top-down hourglass module."""

    def __init__(self, depth: int, channels: int) -> None:
        """Create a recursive hourglass.

        Parameters
        ----------
        depth
            Number of downsampling levels.
        channels
            Feature channel count.
        """
        super().__init__()
        self.upper = ResidualBlock(channels, channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.lower1 = ResidualBlock(channels, channels)
        self.lower2 = (
            Hourglass(depth - 1, channels) if depth > 1 else ResidualBlock(channels, channels)
        )
        self.lower3 = ResidualBlock(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Run hourglass skip fusion.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Multi-scale fused feature map.
        """
        up = self.upper(x)
        low = self.lower3(self.lower2(self.lower1(self.pool(x))))
        low = F.interpolate(low, size=up.shape[-2:], mode="nearest")
        return up + low


class StackedHourglassNet(nn.Module):
    """Compact stacked-hourglass heatmap predictor."""

    def __init__(self, stacks: int = 8, channels: int = 16, joints: int = 16) -> None:
        """Create the stacked hourglass network.

        Parameters
        ----------
        stacks
            Number of hourglass stacks.
        channels
            Feature channel count.
        joints
            Number of heatmap channels.
        """
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            ResidualBlock(channels, channels),
            nn.MaxPool2d(2, 2),
            ResidualBlock(channels, channels),
        )
        self.hourglasses = nn.ModuleList([Hourglass(1, channels) for _ in range(stacks)])
        self.features = nn.ModuleList([ResidualBlock(channels, channels) for _ in range(stacks)])
        self.to_heatmap = nn.ModuleList([nn.Conv2d(channels, joints, 1) for _ in range(stacks)])
        self.merge_features = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1) for _ in range(stacks - 1)]
        )
        self.merge_preds = nn.ModuleList(
            [nn.Conv2d(joints, channels, 1) for _ in range(stacks - 1)]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Predict intermediate pose heatmaps.

        Parameters
        ----------
        x
            Input image batch.

        Returns
        -------
        tuple[Tensor, ...]
            Heatmap predictions, one per stack.
        """
        feat = self.pre(x)
        preds = []
        for idx, hourglass in enumerate(self.hourglasses):
            y = self.features[idx](hourglass(feat))
            pred = self.to_heatmap[idx](y)
            preds.append(pred)
            if idx + 1 < len(self.hourglasses):
                feat = feat + self.merge_features[idx](y) + self.merge_preds[idx](pred)
        return tuple(preds)


def build() -> nn.Module:
    """Build the compact stacked-hourglass network.

    Returns
    -------
    nn.Module
        Random-init stacked-hourglass model.
    """
    return StackedHourglassNet().eval()


def example_input() -> Tensor:
    """Return a pose-estimation image input.

    Returns
    -------
    Tensor
        Example image batch.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = [
    ("StackedHourglass-8stack", "build", "example_input", "2016", "E7"),
]
