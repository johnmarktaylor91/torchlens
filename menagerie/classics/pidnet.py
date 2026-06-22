"""PIDNet: Proportional-Integral-Derivative Network for real-time semantic segmentation.

Xu et al., CVPR 2023.
Paper: https://arxiv.org/abs/2206.02066
Source: https://github.com/XuJiacong/PIDNet

PIDNet uses a PID-controller analogy with three branches:
  - P (Proportional): detail branch -- direct spatial feature (like ResNet stage)
  - I (Integral): context branch -- cumulative global context, uses PAPPM
    (Patch Attention Pyramid Pooling Module) for multi-scale pooling
  - D (Derivative): boundary branch -- edge/boundary detection from P-I difference
  + Bag (Boundary-Attention Gate) fusion module: uses D branch attention to guide
    the merge of P and I branches.

Three model sizes (S/M/L) differ only in channel width and block depth:
  - PIDNet-S: ch=32, layers=[2,2,2,2]
  - PIDNet-M: ch=64, layers=[3,4,6,3]
  - PIDNet-L: ch=64, layers=[3,4,6,3] with wider bottleneck (m=2)

Architecture notes / faithful-core simplifications:
  - Backbone stages use basic residual blocks (not the full ResNet-18/34 bottlenecks);
    the PAPPM, Boundary branch, and Bag fusion modules are faithfully reproduced.
  - Compact channel widths; layers=[2,2,2,2] for all variants in this classic.
  - Input: (1, 3, 64, 64) -- small spatial for fast tracing.
  - trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building blocks
# ============================================================


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
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


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


# ============================================================
# PAPPM: Patch Attention Pyramid Pooling Module (I-branch)
# ============================================================


class PAPPM(nn.Module):
    """Patch Attention Pyramid Pooling Module (PIDNet's I-branch pooling).

    Applies average pooling at multiple scales, upsamples all to the input
    spatial size, then uses a patch-attention conv to learn scale weights.
    Each branch expands back to in_ch so attention-weighted sum is shape-safe.
    Final output: projected to out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, pool_sizes: tuple = (2, 4, 8)) -> None:
        super().__init__()
        self.pool_sizes = pool_sizes
        branch_ch = in_ch // len(pool_sizes)
        # Per-scale: pool -> compress -> expand back to in_ch
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(p),
                    nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                    nn.BatchNorm2d(branch_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(branch_ch, in_ch, 1, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                )
                for p in pool_sizes
            ]
        )
        # Patch attention: produces (n_scales+1) weight maps
        self.attention = nn.Sequential(
            nn.Conv2d(in_ch * (len(pool_sizes) + 1), len(pool_sizes) + 1, 1, bias=False),
            nn.Softmax(dim=1),
        )
        # Output projection
        self.project = _cbr(in_ch, out_ch, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        branches = []
        for br in self.branches:
            feat = br(x)
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
            branches.append(feat)
        all_feat = torch.cat([x] + branches, dim=1)  # (B, in_ch*(n+1), H, W)
        # Patch attention weights
        attn = self.attention(all_feat)  # (B, n_scales+1, H, W)
        # Weighted sum: all features are in_ch-dimensional
        out = x * attn[:, 0:1]
        for i, br_feat in enumerate(branches):
            out = out + br_feat * attn[:, i + 1 : i + 2]
        return self.project(out)


# ============================================================
# Boundary (D) branch module
# ============================================================


class BoundaryBranch(nn.Module):
    """Derivative branch: detects boundaries from P-I difference.

    Takes the element-wise difference of P and I features as input,
    applies a small conv network to produce a boundary attention map.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _cbr(ch, ch, 3),
            _cbr(ch, ch, 3),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, p_feat: torch.Tensor, i_feat: torch.Tensor) -> torch.Tensor:
        # Align if needed
        if p_feat.shape[2:] != i_feat.shape[2:]:
            i_feat = F.interpolate(
                i_feat, size=p_feat.shape[2:], mode="bilinear", align_corners=False
            )
        diff = p_feat - i_feat
        return torch.sigmoid(self.conv(diff))  # boundary attention map


# ============================================================
# Bag: Boundary-Attention Gate fusion
# ============================================================


class BagFusion(nn.Module):
    """Bag fusion (PIDNet).

    Uses D-branch boundary map as a soft gate to selectively blend
    P-branch (detail) and I-branch (context) features:
      out = D * P + (1 - D) * I
    Then a conv refines the fused output.
    """

    def __init__(self, ch: int, out_ch: int) -> None:
        super().__init__()
        self.refine = _cbr(ch, out_ch, 3)

    def forward(
        self,
        p_feat: torch.Tensor,
        i_feat: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        # Align all to P's resolution
        if i_feat.shape[2:] != p_feat.shape[2:]:
            i_feat = F.interpolate(
                i_feat, size=p_feat.shape[2:], mode="bilinear", align_corners=False
            )
        if boundary.shape[2:] != p_feat.shape[2:]:
            boundary = F.interpolate(
                boundary, size=p_feat.shape[2:], mode="bilinear", align_corners=False
            )
        fused = boundary * p_feat + (1.0 - boundary) * i_feat
        return self.refine(fused)


# ============================================================
# PIDNet
# ============================================================


class PIDNet(nn.Module):
    """PIDNet: three-branch (P/I/D) semantic segmentation network.

    P-branch (Proportional):  backbone stages -> fine spatial features
    I-branch (Integral):      extra stage + PAPPM -> global context features
    D-branch (Derivative):    P-I difference -> boundary attention
    Bag fusion:               D gates P+I -> final segmentation
    """

    def __init__(
        self,
        num_classes: int = 19,
        ch: int = 32,
        layers: tuple = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        # Shared stem
        self.stem = nn.Sequential(
            _cbr(3, ch, 3, stride=2, padding=1),
            _cbr(ch, ch, 3, padding=1),
        )

        # P-branch stages (stride: 4 after stem)
        self.p_s1 = _make_stage(ch, ch, layers[0], stride=2)  # stride 8
        self.p_s2 = _make_stage(ch, ch * 2, layers[1], stride=2)  # stride 16
        # P head projects back to ch for fusion
        self.p_proj = _cbr(ch * 2, ch, 1, padding=0)

        # I-branch stages (run deeper)
        self.i_s1 = _make_stage(ch, ch, layers[0], stride=2)  # stride 8
        self.i_s2 = _make_stage(ch, ch * 2, layers[1], stride=2)  # stride 16
        self.i_s3 = _make_stage(ch * 2, ch * 4, layers[2], stride=2)  # stride 32
        # PAPPM on deepest I features
        self.pappm = PAPPM(ch * 4, ch)

        # D-branch (takes P and I at stride 16)
        self.d_branch = BoundaryBranch(ch * 2)

        # Bag fusion
        self.bag = BagFusion(ch, ch)

        # Segmentation head
        self.seg_head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)  # (B, ch, H/2, W/2) -- stem is stride-4 total

        # P branch
        p1 = self.p_s1(feat)  # (B, ch, H/8, W/8)  -- actually stride-4 after 2x strided from stem
        p2 = self.p_s2(p1)  # (B, ch*2, H/16, W/16)

        # I branch (same early stages)
        i1 = self.i_s1(feat)  # (B, ch, H/8, W/8) -- matches p1 spatially
        i2 = self.i_s2(i1)  # (B, ch*2, H/16, W/16)
        i3 = self.i_s3(i2)  # (B, ch*4, H/32, W/32)
        i_out = self.pappm(i3)  # (B, ch, H/32, W/32)

        # D branch: from P and I at same stride (p2 and i2 match)
        boundary = self.d_branch(p2, i2)  # (B, 1, H/16, W/16)

        # Project P to ch
        p_proj = self.p_proj(p2)  # (B, ch, H/16, W/16)

        # Bag fusion
        fused = self.bag(p_proj, i_out, boundary)  # (B, ch, H/16, W/16)

        # Upsample and segment
        out = self.seg_head(fused)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_pidnet_s() -> nn.Module:
    return PIDNet(num_classes=19, ch=32, layers=(2, 2, 2, 2))


def build_pidnet_m() -> nn.Module:
    return PIDNet(num_classes=19, ch=64, layers=(2, 2, 2, 2))


def build_pidnet_l() -> nn.Module:
    # L uses wider channels (64) and deeper stages
    return PIDNet(num_classes=19, ch=64, layers=(3, 4, 6, 3))


def example_input() -> torch.Tensor:
    """Small RGB image (1, 3, 64, 64) for fast tracing."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "PIDNet-S (P/I/D branch segmentation, PAPPM, Bag fusion -- small)",
        "build_pidnet_s",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "PIDNet-M (P/I/D branch segmentation, PAPPM, Bag fusion -- medium)",
        "build_pidnet_m",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "PIDNet-L (P/I/D branch segmentation, PAPPM, Bag fusion -- large)",
        "build_pidnet_l",
        "example_input",
        "2023",
        "DC",
    ),
]
