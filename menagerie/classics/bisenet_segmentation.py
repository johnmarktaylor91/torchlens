"""BiSeNet V1 and V2: Bilateral Segmentation Networks for real-time semantic segmentation.

BiSeNet V1 (Yu et al., ECCV 2018):
  Paper: https://arxiv.org/abs/1808.00897
  Source: https://github.com/ycszen/TorchSeg

  Two-branch architecture:
    - Spatial Path (SP): 3 strided convs -> stride-8 detail features (wide receptive field)
    - Context Path (CP): lightweight backbone + ARM (Attention Refinement Module) at
      tail stages + global avg pooling CIR + FFM (Feature Fusion Module) to merge SP+CP

BiSeNet V2 (Yu et al., IJCV 2021):
  Paper: https://arxiv.org/abs/2004.02147
  Source: https://github.com/MichaelFan01/STDC-Seg (reference implementation)

  Also two branches but with distinct, redesigned primitives:
    - Detail Branch: 3 stages of strided conv blocks preserving fine spatial detail
    - Semantic Branch: lightweight stages with GE (Gather-Expansion) layers and
      CE (Context Embedding) block for global context
    - BGA (Bilateral Guided Aggregation): learnable cross-branch guidance fusion

Architecture notes / faithful-core simplifications:
  - Backbone replaced by a compact 3-stage CNN (ResNet-stub with reduced channels);
    the ARM/FFM/GE/CE/BGA modules are reproduced faithfully from the papers.
  - Compact channel widths (32-64) keep the graph renderable; published widths are 128-256.
  - Input: (1, 3, 64, 128) -- small spatial resolution, batch 1.
  - trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared building blocks
# ============================================================


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# BiSeNet V1 modules
# ============================================================


class ARM(nn.Module):
    """Attention Refinement Module (BiSeNet V1).

    Global average pool -> 1x1 conv -> BN -> sigmoid -> channel-wise multiply.
    Refines the context features by learning an attention mask.
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.bn(self.conv(self.gap(x))))
        return x * attn


class FFM(nn.Module):
    """Feature Fusion Module (BiSeNet V1).

    Concatenates spatial + context features, applies channel attention (SE-style)
    to produce the fused output.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _cbr(in_ch, out_ch)
        self.se_gap = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(out_ch, out_ch // 4, 1, bias=False)
        self.se_fc2 = nn.Conv2d(out_ch // 4, out_ch, 1, bias=False)

    def forward(self, sp: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        # Align spatial sizes
        if sp.shape[2:] != cp.shape[2:]:
            cp = F.interpolate(cp, size=sp.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([sp, cp], dim=1)
        x = self.conv(x)
        attn = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(self.se_gap(x)))))
        return x + x * attn


class _CompactBackbone(nn.Module):
    """Compact 3-stage backbone for the Context Path (stand-in for Xception/ResNet-18).

    Returns (c3, c4) feature maps at strides 16 and 32 relative to input.
    Simplified from published; the ARM+FFM contribution is faithfully reproduced.
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.stem = _cbr(3, base, 3, stride=2, padding=1)  # stride 2
        self.s1 = _cbr(base, base * 2, 3, stride=2, padding=1)  # stride 4
        self.s2 = _cbr(base * 2, base * 4, 3, stride=2, padding=1)  # stride 8
        self.s3 = _cbr(base * 4, base * 8, 3, stride=2, padding=1)  # stride 16
        self.s4 = _cbr(base * 8, base * 8, 3, stride=2, padding=1)  # stride 32

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        c3 = self.s3(x)  # stride 16
        c4 = self.s4(c3)  # stride 32
        return c3, c4


class BiSeNetV1(nn.Module):
    """BiSeNet V1: Spatial Path + Context Path with ARM + FFM.

    Spatial Path: 3 strided convs -> stride-8 high-resolution detail features.
    Context Path: compact backbone -> ARM on c3 and c4 -> upsample and add ->
        global avg pool CIR fused into tail -> FFM merges SP + CP.
    Output: per-pixel class logits at stride 8.
    """

    def __init__(self, num_classes: int = 19, base: int = 32) -> None:
        super().__init__()
        # Spatial path: 3 strided convs to get stride-8 feature
        sp_ch = base * 4  # 128 in paper; we use base*4
        self.sp_conv1 = _cbr(3, base, 3, stride=2, padding=1)
        self.sp_conv2 = _cbr(base, base * 2, 3, stride=2, padding=1)
        self.sp_conv3 = _cbr(base * 2, sp_ch, 3, stride=2, padding=1)

        # Context path
        self.cp_backbone = _CompactBackbone(base)
        c3_ch = base * 8
        c4_ch = base * 8
        self.arm3 = ARM(c3_ch)
        self.arm4 = ARM(c4_ch)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = _cbr(c4_ch, c4_ch, 1, stride=1, padding=0)

        # Upsample tail: c4 + c3 -> cp_ch
        cp_ch = c3_ch
        self.up_conv = _cbr(c4_ch, cp_ch, 3, padding=1)

        # FFM: sp_ch + cp_ch -> sp_ch
        self.ffm = FFM(sp_ch + cp_ch, sp_ch)

        # Segmentation head
        self.seg_head = nn.Sequential(
            _cbr(sp_ch, sp_ch // 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(sp_ch // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial path
        sp = self.sp_conv3(self.sp_conv2(self.sp_conv1(x)))  # stride 8

        # Context path
        c3, c4 = self.cp_backbone(x)
        # Global context injected into c4 tail
        tail = self.gap_conv(self.global_avg_pool(c4))  # (B, c4_ch, 1, 1)
        arm4 = self.arm4(c4) + tail  # broadcast
        arm4_up = F.interpolate(arm4, size=c3.shape[2:], mode="bilinear", align_corners=False)
        arm3 = self.arm3(c3) + arm4_up
        cp = self.up_conv(arm3)
        cp_up = F.interpolate(cp, size=sp.shape[2:], mode="bilinear", align_corners=False)

        # Fusion
        out = self.ffm(sp, cp_up)
        out = self.seg_head(out)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ============================================================
# BiSeNet V2 modules
# ============================================================


class _GELayer(nn.Module):
    """Gather-Expansion Layer (BiSeNet V2).

    Stride-1: 3x3 dw -> expand -> 3x3 dw (dw=depthwise) -> project
    Stride-2: adds an extra strided 3x3 dw and a shortcut strided avg pool.
    Faithfully reproduces the GE topology from Figure 4 of the paper.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 6) -> None:
        super().__init__()
        mid_ch = in_ch * expand
        if stride == 1:
            self.conv = nn.Sequential(
                _cbr(in_ch, in_ch, 3, stride=1, padding=1),
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.shortcut = (
                nn.Identity()
                if in_ch == out_ch
                else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            )
        else:
            self.conv = nn.Sequential(
                _cbr(in_ch, in_ch, 3, stride=1, padding=1),
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class _CEBlock(nn.Module):
    """Context Embedding Block (BiSeNet V2).

    Global average pool -> BN -> 1x1 conv -> add back (broadcast) -> 3x3 conv.
    Embeds global context into every spatial location.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.conv1(self.bn(self.gap(x)))  # (B, ch, 1, 1)
        return self.conv2(x + ctx)


class BGA(nn.Module):
    """Bilateral Guided Aggregation (BiSeNet V2).

    Detail branch -> spatial attention guide -> modulates semantic branch.
    Semantic branch -> channel attention guide -> modulates detail branch.
    Both guided outputs summed -> fused output.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        # Detail -> guide semantic
        self.d_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.d_pw = nn.Conv2d(ch, ch, 1, bias=False)
        # Semantic -> guide detail
        self.s_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.s_pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.s_avg = nn.AvgPool2d(3, stride=1, padding=1)
        # Output project
        self.out_conv = _cbr(ch, ch)

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        # Align semantic to detail resolution
        if detail.shape[2:] != semantic.shape[2:]:
            semantic_up = F.interpolate(
                semantic, size=detail.shape[2:], mode="bilinear", align_corners=False
            )
        else:
            semantic_up = semantic

        # Detail branch guide: sigmoid of DW+PW on detail
        d_attn = torch.sigmoid(self.d_pw(self.d_dw(detail)))
        s_guided = semantic_up * d_attn

        # Semantic branch guide: sigmoid of DW+PW on semantic_up
        s_attn = torch.sigmoid(self.s_dw(self.s_avg(semantic_up)))
        d_guided = detail * s_attn
        # project s through another path
        s_proj = self.s_pw(semantic_up)

        fused = d_guided + s_proj * s_attn + s_guided
        return self.out_conv(fused)


class BiSeNetV2(nn.Module):
    """BiSeNet V2: Detail Branch + Semantic Branch + BGA fusion.

    Detail Branch: 3 stages of strided conv blocks preserving spatial detail.
    Semantic Branch: stem + S1-S5 stages with GE layers + CE context block.
    BGA: bilateral guided aggregation of both branches.
    Output: per-pixel class logits upsampled to input resolution.
    """

    def __init__(self, num_classes: int = 19, ch: int = 32) -> None:
        super().__init__()
        # Detail branch: 3 stages, strides 2/2/1
        self.detail_s1 = nn.Sequential(_cbr(3, ch, 3, stride=2, padding=1))
        self.detail_s2 = nn.Sequential(
            _cbr(ch, ch, 3, stride=2, padding=1),
            _cbr(ch, ch, 3),
        )
        self.detail_s3 = nn.Sequential(
            _cbr(ch, ch * 2, 3, stride=2, padding=1),
            _cbr(ch * 2, ch * 2, 3),
        )
        detail_out_ch = ch * 2  # stride 8

        # Semantic branch
        sem_ch = ch
        self.sem_stem = _cbr(3, sem_ch, 3, stride=4, padding=1)  # stride 4
        self.sem_s3 = _GELayer(sem_ch, sem_ch * 2, stride=2)  # stride 8
        self.sem_s4 = nn.Sequential(
            _GELayer(sem_ch * 2, sem_ch * 4, stride=2),  # stride 16
            _GELayer(sem_ch * 4, sem_ch * 4, stride=1),
        )
        self.sem_s5 = nn.Sequential(
            _GELayer(sem_ch * 4, sem_ch * 8, stride=2),  # stride 32
            _GELayer(sem_ch * 8, sem_ch * 8, stride=1),
            _GELayer(sem_ch * 8, sem_ch * 8, stride=1),
            _GELayer(sem_ch * 8, sem_ch * 8, stride=1),
        )
        self.ce = _CEBlock(sem_ch * 8)
        # Project semantic to match detail_out_ch for BGA
        self.sem_proj = _cbr(sem_ch * 8, detail_out_ch, 1, padding=0)

        # BGA
        self.bga = BGA(detail_out_ch)

        # Seg head
        self.seg_head = nn.Sequential(
            _cbr(detail_out_ch, detail_out_ch),
            nn.Conv2d(detail_out_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detail branch
        d = self.detail_s1(x)
        d = self.detail_s2(d)
        d = self.detail_s3(d)  # (B, ch*2, H/8, W/8)

        # Semantic branch
        s = self.sem_stem(x)
        s = self.sem_s3(s)
        s = self.sem_s4(s)
        s = self.sem_s5(s)
        s = self.ce(s)
        s = self.sem_proj(s)  # (B, ch*2, H/32, W/32) -> projected

        # BGA
        fused = self.bga(d, s)  # (B, ch*2, H/8, W/8)

        # Segmentation output
        out = self.seg_head(fused)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_bisenet_v1() -> nn.Module:
    return BiSeNetV1(num_classes=19, base=32)


def build_bisenet_v2() -> nn.Module:
    return BiSeNetV2(num_classes=19, ch=32)


def example_input() -> torch.Tensor:
    """Small RGB image (1, 3, 64, 128) for fast tracing."""
    return torch.randn(1, 3, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "BiSeNetV1 (spatial path + context path + ARM + FFM)",
        "build_bisenet_v1",
        "example_input",
        "2018",
        "DC",
    ),
    (
        "BiSeNetV2 (detail branch + GE semantic branch + BGA)",
        "build_bisenet_v2",
        "example_input",
        "2021",
        "DC",
    ),
]
