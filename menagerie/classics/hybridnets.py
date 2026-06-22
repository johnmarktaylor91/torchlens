"""HybridNets: End-to-End Perception Network for Multi-Task Driving Perception.

Vu et al., 2022.
Paper: https://arxiv.org/abs/2203.09035
Source: https://github.com/datvuthanh/HybridNets

HybridNets is an end-to-end multi-task driving perception network combining:
  1. EfficientNet-B0 backbone (efficient scalable CNN with compound scaling)
  2. BiFPN neck (bidirectional feature pyramid network -- weighted multi-scale fusion)
  3. Three task heads:
     - Object detection head: anchor-based single-stage detector (classification + box regression)
     - Drivable area segmentation head: simple upsampling decoder producing binary drivable mask
     - Lane line segmentation head: upsampling decoder producing lane binary mask

Key architecture primitives faithfully reproduced:
  - BiFPN: bidirectional feature pyramid with fast-normalized fusion (epsilon-normalized
    learned weights summing top-down and bottom-up paths at each scale)
  - Three-head multi-task topology
  - EfficientNet-style compound-scaled backbone (B0 profile: width=1.0, depth=1.0)

Architecture notes / faithful-core simplifications:
  - EfficientNet-B0 simplified to a 3-stage MBConv backbone (inverted residual blocks)
    with squeeze-and-excite; channel widths reduced proportionally.
  - BiFPN: 3 feature scales, 2 BiFPN rounds (published: 3 rounds), faithfully bidirectional.
  - Detection head: class + box regression applied at each BiFPN level (anchor-per-cell design).
  - Input: (1, 3, 64, 128) -- small for fast tracing.
  - trace+draw verified 2026-06-21.
  - Covers both HybridNets and HybridNets-B0 target names (same architecture, B0 backbone config).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EfficientNet-B0 style backbone (compact MBConv + SE)
# ============================================================


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class SEBlock(nn.Module):
    def __init__(self, ch: int, ratio: int = 4) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, max(1, ch // ratio), 1, bias=True)
        self.fc2 = nn.Conv2d(max(1, ch // ratio), ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(self.gap(x)))))
        return x * s


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv with SE."""

    def __init__(self, in_ch: int, out_ch: int, expand: int = 6, stride: int = 1) -> None:
        super().__init__()
        mid = in_ch * expand
        self.expand = (
            nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(inplace=True),
            )
            if expand > 1
            else nn.Identity()
        )
        exp_ch = mid if expand > 1 else in_ch
        self.dw = nn.Sequential(
            nn.Conv2d(exp_ch, exp_ch, 3, stride=stride, padding=1, groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            nn.SiLU(inplace=True),
        )
        self.se = SEBlock(exp_ch)
        self.project = nn.Sequential(
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_skip = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x) if not isinstance(self.expand, nn.Identity) else x
        h = self.dw(h)
        h = self.se(h)
        h = self.project(h)
        return h + x if self.use_skip else h


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0-style backbone returning 3 feature scales (C3/C4/C5).

    Compact channel widths for fast tracing; topology matches B0 stage layout.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        # Stem
        self.stem = _cbr(3, base, 3, stride=2, padding=1)
        # Stage 1: stride-1 MBConvs
        self.stage1 = nn.Sequential(MBConv(base, base, expand=1, stride=1))
        # Stage 2: stride-2, doubles channels
        self.stage2 = nn.Sequential(
            MBConv(base, base * 2, expand=6, stride=2),
            MBConv(base * 2, base * 2, expand=6, stride=1),
        )
        # Stage 3: stride-2
        self.stage3 = nn.Sequential(
            MBConv(base * 2, base * 4, expand=6, stride=2),
            MBConv(base * 4, base * 4, expand=6, stride=1),
        )
        # Stage 4: stride-2 (C5)
        self.stage4 = nn.Sequential(
            MBConv(base * 4, base * 8, expand=6, stride=2),
            MBConv(base * 8, base * 8, expand=6, stride=1),
        )
        self.c3_ch = base * 2  # stride 8
        self.c4_ch = base * 4  # stride 16
        self.c5_ch = base * 8  # stride 32

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)  # stride 8
        c4 = self.stage3(c3)  # stride 16
        c5 = self.stage4(c4)  # stride 32
        return c3, c4, c5


# ============================================================
# BiFPN: Bidirectional Feature Pyramid Network
# ============================================================


class BiFPNLayer(nn.Module):
    """Single BiFPN layer: top-down + bottom-up with fast-normalized fusion.

    Fast normalized fusion: w_i = ReLU(w_i) / (sum_j ReLU(w_j) + eps)
    Applied at each scale during both top-down and bottom-up passes.
    """

    def __init__(self, feat_ch: int, n_scales: int = 3, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.n_scales = n_scales
        # Top-down fusion weights (2 inputs per scale except highest)
        self.td_w = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(n_scales - 1)])
        # Bottom-up fusion weights (3 inputs for middle scales, 2 for bottom)
        bu_n = [3 if (i > 0 and i < n_scales - 1) else 2 for i in range(n_scales)]
        self.bu_w = nn.ParameterList([nn.Parameter(torch.ones(n)) for n in bu_n])
        # Depthwise-separable conv per scale (after each fusion)
        self.td_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feat_ch, feat_ch, 3, padding=1, groups=feat_ch, bias=False),
                    nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
                    nn.BatchNorm2d(feat_ch),
                    nn.SiLU(inplace=True),
                )
                for _ in range(n_scales - 1)
            ]
        )
        self.bu_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feat_ch, feat_ch, 3, padding=1, groups=feat_ch, bias=False),
                    nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
                    nn.BatchNorm2d(feat_ch),
                    nn.SiLU(inplace=True),
                )
                for _ in range(n_scales)
            ]
        )

    @staticmethod
    def _fuse(feats, weights, eps):
        w = F.relu(torch.stack([weights[i] for i in range(len(feats))]))
        w = w / (w.sum() + eps)
        out = sum(f * w[i] for i, f in enumerate(feats))
        return out

    def forward(self, features):
        """
        features: list of (B, feat_ch, H_i, W_i) at each scale (fine to coarse)
        Returns updated features list.
        """
        n = self.n_scales
        # Top-down path (from coarsest scale down to finest)
        td = [None] * n
        td[n - 1] = features[n - 1]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=features[i].shape[2:], mode="nearest")
            w = F.relu(self.td_w[i])
            w = w / (w.sum() + self.eps)
            fused = w[0] * features[i] + w[1] * up
            td[i] = self.td_convs[i](fused)

        # Bottom-up path (from finest back to coarsest)
        out = [None] * n
        out[0] = self.bu_convs[0](self._fuse([features[0], td[0]], self.bu_w[0], self.eps))
        for i in range(1, n - 1):
            down = F.adaptive_avg_pool2d(out[i - 1], output_size=td[i].shape[2:])
            w = F.relu(self.bu_w[i])
            w = w / (w.sum() + self.eps)
            fused = w[0] * features[i] + w[1] * td[i] + w[2] * down
            out[i] = self.bu_convs[i](fused)
        # Coarsest scale
        down = F.adaptive_avg_pool2d(out[n - 2], output_size=features[n - 1].shape[2:])
        out[n - 1] = self.bu_convs[n - 1](
            self._fuse([features[n - 1], down], self.bu_w[n - 1], self.eps)
        )
        return out


class BiFPN(nn.Module):
    """BiFPN: repeated BiFPN layers + scale projection to uniform feat_ch."""

    def __init__(
        self,
        in_channels,  # list of input channel counts per scale
        feat_ch: int = 64,
        n_rounds: int = 2,
        n_scales: int = 3,
    ) -> None:
        super().__init__()
        # Project each input scale to feat_ch
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, feat_ch, 1, bias=False),
                    nn.BatchNorm2d(feat_ch),
                )
                for c in in_channels
            ]
        )
        self.layers = nn.Sequential(*[BiFPNLayer(feat_ch, n_scales) for _ in range(n_rounds)])

    def forward(self, features):
        feats = [self.proj[i](features[i]) for i in range(len(features))]
        for layer in self.layers:
            feats = layer(feats)
        return feats


# ============================================================
# Task heads
# ============================================================


class DetectionHead(nn.Module):
    """Anchor-based detection head (1 anchor per cell for simplicity).

    Outputs class logits and box regression at each BiFPN scale level.
    """

    def __init__(self, feat_ch: int, num_classes: int = 2, n_levels: int = 3) -> None:
        super().__init__()
        self.cls_heads = nn.ModuleList(
            [nn.Conv2d(feat_ch, num_classes, 3, padding=1) for _ in range(n_levels)]
        )
        self.box_heads = nn.ModuleList(
            [nn.Conv2d(feat_ch, 4, 3, padding=1) for _ in range(n_levels)]
        )

    def forward(self, feats):
        cls_outs, box_outs = [], []
        for i, f in enumerate(feats):
            cls_outs.append(self.cls_heads[i](f))
            box_outs.append(self.box_heads[i](f))
        return cls_outs, box_outs


class SegHead(nn.Module):
    """Simple upsampling segmentation head (drivable area or lane)."""

    def __init__(self, feat_ch: int, num_classes: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, num_classes, 1),
        )

    def forward(self, feat: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.conv(feat)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


# ============================================================
# Full HybridNets
# ============================================================


class HybridNets(nn.Module):
    """HybridNets: EfficientNet-B0 backbone + BiFPN neck + 3 task heads.

    Three simultaneous outputs:
      - det_cls, det_box: object detection logits at 3 BiFPN scales
      - drive_seg: drivable area segmentation mask
      - lane_seg: lane line segmentation mask
    """

    def __init__(
        self,
        num_det_classes: int = 2,
        base: int = 16,
        bifpn_ch: int = 48,
        bifpn_rounds: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = EfficientNetB0Backbone(base)
        in_channels = [
            self.backbone.c3_ch,
            self.backbone.c4_ch,
            self.backbone.c5_ch,
        ]
        self.bifpn = BiFPN(in_channels, feat_ch=bifpn_ch, n_rounds=bifpn_rounds, n_scales=3)
        self.det_head = DetectionHead(bifpn_ch, num_det_classes, n_levels=3)
        self.drive_seg_head = SegHead(bifpn_ch, num_classes=1)
        self.lane_seg_head = SegHead(bifpn_ch, num_classes=2)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]
        c3, c4, c5 = self.backbone(x)
        feats = self.bifpn([c3, c4, c5])
        # Detection at all scales
        det_cls, det_box = self.det_head(feats)
        # Seg heads use the finest-scale BiFPN feature
        drive_seg = self.drive_seg_head(feats[0], (H, W))
        lane_seg = self.lane_seg_head(feats[0], (H, W))
        return det_cls, det_box, drive_seg, lane_seg


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_hybridnets() -> nn.Module:
    return HybridNets(num_det_classes=2, base=16, bifpn_ch=48, bifpn_rounds=2)


def build_hybridnets_b0() -> nn.Module:
    """HybridNets with B0-profile backbone (same as generic; explicit B0 entry)."""
    return HybridNets(num_det_classes=2, base=16, bifpn_ch=48, bifpn_rounds=2)


def example_input() -> torch.Tensor:
    """RGB image (1, 3, 64, 128) for fast tracing."""
    return torch.randn(1, 3, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "HybridNets (EfficientNet-B0 + BiFPN + det/drivable/lane multi-task heads)",
        "build_hybridnets",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "HybridNets-B0 (EfficientNet-B0 backbone; same as HybridNets, explicit B0 entry)",
        "build_hybridnets_b0",
        "example_input",
        "2022",
        "DC",
    ),
]
