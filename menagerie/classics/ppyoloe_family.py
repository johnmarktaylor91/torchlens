"""PP-YOLOE and PP-YOLOE+ compact detectors.

PP-YOLOE is PaddleDetection's evolved YOLO detector: CSPRepResStage backbone,
PAN/FPN neck, anchor-free decoupled ET-head, and TAL-style deployment-friendly
classification/regression outputs.  The compact models below preserve those
structural pieces with s/m/l/x and plus variants scaled by depth and width.

Source: "PP-YOLOE: An evolved version of YOLO" (arXiv:2203.16250).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ScaleName = Literal["s", "m", "l", "x"]


class RepBlock(nn.Module):
    """Rep-style convolution block used inside CSPRep stages.

    Parameters
    ----------
    channels:
        Channel count.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel 3x3/1x1 rep branches.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        return F.silu(self.conv3(x) + self.conv1(x) + x)


class CSPRepStage(nn.Module):
    """CSPRepResStage-style split-transform-merge stage.

    Parameters
    ----------
    channels:
        Channel count.
    depth:
        Number of rep blocks.
    """

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.left = nn.Conv2d(channels, channels // 2, 1)
        self.right = nn.Conv2d(channels, channels // 2, 1)
        self.blocks = nn.ModuleList([RepBlock(channels // 2) for _ in range(depth)])
        self.merge = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a CSPRep stage.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Stage output.
        """

        left = self.left(x)
        for block in self.blocks:
            left = block(left)
        right = self.right(x)
        return F.silu(self.merge(torch.cat((left, right), dim=1)))


class PPYOLOECompact(nn.Module):
    """Compact PP-YOLOE detector.

    Parameters
    ----------
    scale:
        s/m/l/x scale key.
    plus:
        Whether to use the deeper PP-YOLOE+ variant.
    num_classes:
        Number of detection classes.
    """

    def __init__(self, scale: ScaleName, plus: bool = False, num_classes: int = 5) -> None:
        super().__init__()
        widths = {"s": 24, "m": 32, "l": 40, "x": 48}
        depths = {"s": 1, "m": 2, "l": 3, "x": 4}
        width = widths[scale]
        depth = depths[scale] + int(plus)
        self.stem = nn.Conv2d(3, width, 3, stride=2, padding=1)
        self.stage1 = CSPRepStage(width, depth)
        self.down1 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.stage2 = CSPRepStage(width, depth)
        self.down2 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.stage3 = CSPRepStage(width, depth)
        self.lateral = nn.Conv2d(width, width, 1)
        self.pan = nn.Conv2d(width * 2, width, 3, padding=1)
        self.cls_head = nn.Conv2d(width, num_classes, 1)
        self.reg_head = nn.Conv2d(width, 4, 1)
        self.obj_head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict anchor-free detection maps.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Flattened ``class + box + objectness`` predictions.
        """

        c3 = self.stage1(F.silu(self.stem(x)))
        c4 = self.stage2(F.silu(self.down1(c3)))
        c5 = self.stage3(F.silu(self.down2(c4)))
        up = F.interpolate(self.lateral(c5), size=c4.shape[-2:], mode="nearest")
        neck = F.silu(self.pan(torch.cat((up, c4), dim=1)))
        cls = self.cls_head(neck)
        reg = F.softplus(self.reg_head(neck))
        obj = self.obj_head(neck)
        return torch.cat((cls, reg, obj), dim=1).flatten(2).transpose(1, 2)


def _build(scale: ScaleName, plus: bool = False) -> PPYOLOECompact:
    """Build a PP-YOLOE variant.

    Parameters
    ----------
    scale:
        Model scale.
    plus:
        Whether to build PP-YOLOE+.

    Returns
    -------
    PPYOLOECompact
        Random-initialized detector.
    """

    return PPYOLOECompact(scale, plus=plus)


def example_input() -> torch.Tensor:
    """Create a compact detector input.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.rand(1, 3, 64, 64)


def build_s() -> PPYOLOECompact:
    """Build PP-YOLOE-S."""

    return _build("s")


def build_m() -> PPYOLOECompact:
    """Build PP-YOLOE-M."""

    return _build("m")


def build_l() -> PPYOLOECompact:
    """Build PP-YOLOE-L."""

    return _build("l")


def build_x() -> PPYOLOECompact:
    """Build PP-YOLOE-X."""

    return _build("x")


def build_plus_s() -> PPYOLOECompact:
    """Build PP-YOLOE+S."""

    return _build("s", plus=True)


def build_plus_m() -> PPYOLOECompact:
    """Build PP-YOLOE+M."""

    return _build("m", plus=True)


def build_plus_l() -> PPYOLOECompact:
    """Build PP-YOLOE+L."""

    return _build("l", plus=True)


def build_plus_x() -> PPYOLOECompact:
    """Build PP-YOLOE+X."""

    return _build("x", plus=True)


MENAGERIE_ENTRIES = [
    ("paddledet_ppyoloe_l", "build_l", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_m", "build_m", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_plus_l", "build_plus_l", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_plus_m", "build_plus_m", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_plus_s", "build_plus_s", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_plus_x", "build_plus_x", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_s", "build_s", "example_input", "2022", "vision/detection"),
    ("paddledet_ppyoloe_x", "build_x", "example_input", "2022", "vision/detection"),
]
