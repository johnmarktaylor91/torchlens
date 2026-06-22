"""BezierLaneNet: End-to-end Lane Detection via Bezier Curve Regression.

Feng et al., CVPR 2022.
Paper: https://arxiv.org/abs/2204.11548
Source: https://github.com/voldemortX/pytorch-auto-drive

Key idea: Instead of segmenting pixels or regressing anchor offsets,
BezierLaneNet directly regresses Bezier control points for each lane.
The distinctive contributions:
  1. Feature Flip Fusion (FFF): flips the feature map horizontally and
     concatenates with the original -- exploits the left-right symmetry
     of roads to enrich lane features.
  2. Bezier-curve regression head: predicts (n_control_points, 2) control
     point coordinates per lane query, enabling smooth curve parameterization.
  3. Row-by-row loss formulation sampling points on predicted curves.

Architecture: backbone CNN -> FFF -> detection head regressing Bezier control points.

Architecture notes / faithful-core simplifications:
  - Backbone: compact 4-stage CNN (ResNet-18 stub), contribution is FFF + Bezier head
  - FFF module faithfully reproduced: flip + concat + project
  - Bezier regression head: per-lane control-point prediction from pooled features
  - Input: (1, 3, 64, 128) -- small for fast tracing
  - trace+draw verified 2026-06-21
  - Covers both BezierLaneNet-ResNet18 and PytorchAutoDrive-BezierLaneNet target names.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Compact backbone
# ============================================================


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt)


class CompactBackbone(nn.Module):
    """ResNet-18-style backbone returning C5 feature map."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(base, base), BasicBlock(base, base))
        self.layer2 = nn.Sequential(
            BasicBlock(base, base * 2, stride=2), BasicBlock(base * 2, base * 2)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(base * 2, base * 4, stride=2), BasicBlock(base * 4, base * 4)
        )
        self.out_ch = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B, out_ch, H/16, W/16)


# ============================================================
# Feature Flip Fusion (FFF) -- BezierLaneNet's distinctive module
# ============================================================


class FeatureFlipFusion(nn.Module):
    """Feature Flip Fusion.

    Flips the feature map along the width axis and concatenates with the
    original, then projects back to the original channel count.
    Exploits road left-right symmetry to improve lane detection.
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.proj = _ConvBNReLU(in_ch * 2, in_ch, k=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flip = torch.flip(x, dims=[3])  # horizontal flip
        fused = torch.cat([x, x_flip], dim=1)  # (B, 2*C, H, W)
        return self.proj(fused)  # (B, C, H, W)


# ============================================================
# Bezier Lane Head
# ============================================================


class BezierLaneHead(nn.Module):
    """Bezier control-point regression head.

    Global average pool + FC -> regress (n_lanes, n_control_points, 2) coords.
    Also outputs a per-lane existence classification score.
    """

    def __init__(self, in_ch: int, n_lanes: int = 4, n_ctrl: int = 4) -> None:
        super().__init__()
        self.n_lanes = n_lanes
        self.n_ctrl = n_ctrl
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = in_ch
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
        )
        # Bezier control points: n_lanes * n_ctrl * 2 (x,y)
        self.ctrl_head = nn.Linear(hidden, n_lanes * n_ctrl * 2)
        # Existence classification
        self.cls_head = nn.Linear(hidden, n_lanes)

    def forward(self, x: torch.Tensor):
        feat = self.fc(self.gap(x))  # (B, hidden)
        ctrl = self.ctrl_head(feat).view(-1, self.n_lanes, self.n_ctrl, 2)  # Bezier pts
        cls = self.cls_head(feat)  # (B, n_lanes)
        return ctrl, cls


# ============================================================
# Full BezierLaneNet
# ============================================================


class BezierLaneNet(nn.Module):
    """BezierLaneNet: backbone + FFF + Bezier regression head."""

    def __init__(self, n_lanes: int = 4, n_ctrl: int = 4, base: int = 32) -> None:
        super().__init__()
        self.backbone = CompactBackbone(base)
        self.fff = FeatureFlipFusion(self.backbone.out_ch)
        self.head = BezierLaneHead(self.backbone.out_ch, n_lanes, n_ctrl)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.fff(feat)
        ctrl_pts, cls_logits = self.head(feat)
        return ctrl_pts, cls_logits


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_bezier_lane_net() -> nn.Module:
    return BezierLaneNet(n_lanes=4, n_ctrl=4, base=32)


def example_input() -> torch.Tensor:
    """RGB image (1, 3, 64, 128) for fast tracing."""
    return torch.randn(1, 3, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "BezierLaneNet (feature-flip-fusion + Bezier control-point lane regression)",
        "build_bezier_lane_net",
        "example_input",
        "2022",
        "DC",
    ),
]
