"""Monocular depth estimation: FCRN, GLPDepth, LapDepth, LeRes, PixelFormer.

FCRN (fcrn_resnet50_depth):
  Laina et al., 3DV 2016. https://arxiv.org/abs/1606.00373
  Source: https://github.com/iro-cp/FCRN-DepthPrediction
  Distinctive: ResNet50 encoder + Up-Projection blocks (up-conv with
  2 parallel branches then element-wise max-pooling) for decoder.

GLPDepth (glpdepth_swin):
  Kim & Kim, ECCV 2022. https://arxiv.org/abs/2201.07436
  Source: https://github.com/vinvino02/GLPDepth
  Distinctive: Hierarchical encoder (Swin/MiT stages with patch merging) +
  lightweight decoder with Selective Feature Fusion (SFF): at each decoder
  step, fuse encoder skip + previous decoder output via learned attention gate.

LapDepth (lapdepth_resnet):
  Song et al., IEEE Trans. Circuits 2021. https://arxiv.org/abs/2105.01016
  Source: https://github.com/tjqansthd/LapDepth-release
  Distinctive: Laplacian pyramid decoder — each level predicts a residual
  Laplacian depth component; coarse depth added to progressively finer residuals.

LeRes (leres / leres_resnext101):
  Wei et al., CVPR 2021. https://arxiv.org/abs/2104.11610
  Source: https://github.com/aim-uofa/AdelaiDepth
  Distinctive: ResNeXt backbone encoder + Attention-based decoder head;
  Pairwise Normal Regression (depth slope estimation) + point-cloud module.
  Two variants: leres (ResNet50) and leres_resnext101 (ResNeXt101).

PixelFormer (pixelformer):
  Agarwal & Arora, ECCV 2022. https://arxiv.org/abs/2208.08635
  Source: https://github.com/ashutosh1807/PixelFormer
  Distinctive: Swin encoder + Skip Attention Module (SAM): cross-attention
  between decoder queries and encoder skip features; bin-center predicting
  transformer decoder head (adaptive bins).

All: compact config, RGB input (1, 3, H, W) with H=64, W=128.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


def _upsample(src: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(src, size=size, mode="bilinear", align_corners=False)


# ══════════════════════════════════════════════════════════════
# FCRN: ResNet50 encoder + Up-Projection decoder
# ══════════════════════════════════════════════════════════════


class _UpProjection(nn.Module):
    """FCRN Up-Projection block: 2 branches after 2x upsample, element-wise max."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        # Branch 1: conv -> BN -> ReLU -> conv -> BN
        self.b1_conv1 = nn.Conv2d(in_c, out_c, 5, 1, 2, bias=False)
        self.b1_bn1 = nn.BatchNorm2d(out_c)
        self.b1_conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.b1_bn2 = nn.BatchNorm2d(out_c)
        # Branch 2: conv -> BN
        self.b2_conv = nn.Conv2d(in_c, out_c, 5, 1, 2, bias=False)
        self.b2_bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        b1 = F.relu(self.b1_bn1(self.b1_conv1(x)))
        b1 = self.b1_bn2(self.b1_conv2(b1))
        b2 = self.b2_bn(self.b2_conv(x))
        return F.relu(torch.max(b1, b2))


class _ResBlock(nn.Module):
    """Compact ResNet block (no bottleneck)."""

    def __init__(self, C: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.skip = (
            nn.Identity()
            if stride == 1
            else nn.Sequential(nn.Conv2d(C, C, 1, stride, bias=False), nn.BatchNorm2d(C))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class FCRN(nn.Module):
    """FCRN: ResNet encoder + Up-Projection decoder for monocular depth."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        # Compact ResNet50-style encoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 7, 2, 3, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(_ResBlock(C), _ResBlock(C))
        self.layer2 = nn.Sequential(
            nn.Conv2d(C, C * 2, 1, 2, bias=False), _ResBlock(C * 2), _ResBlock(C * 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 1, 2, bias=False), _ResBlock(C * 4), _ResBlock(C * 4)
        )
        # Up-Projection decoder
        self.up1 = _UpProjection(C * 4, C * 2)
        self.up2 = _UpProjection(C * 2, C)
        self.up3 = _UpProjection(C, C // 2)
        self.up4 = _UpProjection(C // 2, C // 4)
        self.depth_head = nn.Conv2d(C // 4, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)
        f1 = self.layer1(s)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        d = self.up1(f3)
        d = self.up2(d)
        d = self.up3(d)
        d = self.up4(d)
        d = self.depth_head(d)
        d = F.interpolate(d, x.shape[2:], mode="bilinear", align_corners=False)
        return d


# ══════════════════════════════════════════════════════════════
# GLPDepth: hierarchical encoder + Selective Feature Fusion decoder
# ══════════════════════════════════════════════════════════════


class _HierarchicalEncoder(nn.Module):
    """Compact hierarchical feature extractor (4 stages with patch merging)."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, C, 7, 2, 3, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        # patch merge at each stage (2x downsample)
        self.merge1 = nn.Conv2d(C, C * 2, 2, 2)
        self.stage2 = nn.Sequential(
            _ConvBnRelu(C * 2, C * 2),
            _ConvBnRelu(C * 2, C * 2),
        )
        self.merge2 = nn.Conv2d(C * 2, C * 4, 2, 2)
        self.stage3 = nn.Sequential(
            _ConvBnRelu(C * 4, C * 4),
            _ConvBnRelu(C * 4, C * 4),
        )
        self.merge3 = nn.Conv2d(C * 4, C * 8, 2, 2)
        self.stage4 = nn.Sequential(
            _ConvBnRelu(C * 8, C * 8),
            _ConvBnRelu(C * 8, C * 8),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(self.merge1(f1))
        f3 = self.stage3(self.merge2(f2))
        f4 = self.stage4(self.merge3(f3))
        return [f1, f2, f3, f4]


class _SFF(nn.Module):
    """Selective Feature Fusion: attention gate between skip + decoder features."""

    def __init__(self, skip_c: int, dec_c: int, out_c: int) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_c, out_c, 1)
        self.dec_proj = nn.Conv2d(dec_c, out_c, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, 1),
            nn.Sigmoid(),
        )
        self.out = _ConvBnRelu(out_c, out_c)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        sp = self.skip_proj(skip)
        dp = self.dec_proj(F.interpolate(dec, skip.shape[2:], mode="bilinear", align_corners=False))
        g = self.gate(torch.cat([sp, dp], 1))
        return self.out(g * sp + (1 - g) * dp)


class GLPDepth(nn.Module):
    """GLPDepth: hierarchical encoder + SFF decoder for monocular depth."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.enc = _HierarchicalEncoder(C)
        self.sff3 = _SFF(C * 4, C * 8, C * 4)
        self.sff2 = _SFF(C * 2, C * 4, C * 2)
        self.sff1 = _SFF(C, C * 2, C)
        self.depth_head = nn.Conv2d(C, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.enc(x)
        d = self.sff3(f3, f4)
        d = self.sff2(f2, d)
        d = self.sff1(f1, d)
        d = _upsample(self.depth_head(d), x.shape[2:])
        return d


# ══════════════════════════════════════════════════════════════
# LapDepth: Laplacian pyramid decoder
# ══════════════════════════════════════════════════════════════


class _LapDecLevel(nn.Module):
    """One Laplacian decoder level: predict residual depth component."""

    def __init__(self, in_c: int, C: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBnRelu(in_c, C),
            nn.Conv2d(C, 1, 3, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)


class LapDepth(nn.Module):
    """LapDepth: ResNet encoder + Laplacian-pyramid decoder."""

    def __init__(self, C: int = 16, n_levels: int = 3) -> None:
        super().__init__()
        self.n_levels = n_levels
        # Shared encoder (ResNet-lite)
        self.enc = nn.Sequential(
            nn.Conv2d(3, C, 7, 2, 3, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 4, 3, 2, 1),
        )
        # Laplacian decoder levels (coarse -> fine)
        # Level 0: feat_c = C*4 (no previous depth)
        # Level i>0: feat_c = C*4 + 1 (feat upsampled + previous depth upsampled)
        feat_cs = [C * 4] + [C * 4 + 1] * (n_levels - 1)
        self.lap_levels = nn.ModuleList([_LapDecLevel(fc, C * 2) for fc in feat_cs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)
        depth = None
        for i, level in enumerate(self.lap_levels):
            if depth is None:
                inp = feat
            else:
                depth_up = F.interpolate(
                    depth, feat.shape[2:], mode="bilinear", align_corners=False
                )
                inp = torch.cat([feat, depth_up], dim=1)
            lap = level(inp)
            if depth is None:
                depth = lap
            else:
                depth = (
                    F.interpolate(depth, lap.shape[2:], mode="bilinear", align_corners=False) + lap
                )
            # upsample feat for next level
            if i < self.n_levels - 1:
                feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
        depth = F.interpolate(depth, x.shape[2:], mode="bilinear", align_corners=False)
        return depth


# ══════════════════════════════════════════════════════════════
# LeRes: ResNet/ResNeXt encoder + attention decoder (depth backbone)
# ══════════════════════════════════════════════════════════════


class _ChannelAttention(nn.Module):
    def __init__(self, C: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(C, max(C // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(C // reduction, 4), C),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.shape[0], -1, 1, 1)


class _LeResEncoder(nn.Module):
    """ResNet-style encoder (lite) for LeRes."""

    def __init__(self, C: int = 16, deep: bool = False) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(3, C, 7, 2, 3, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 4, 3, 2, 1),
        ]
        if deep:
            layers += [_ConvBnRelu(C * 4, C * 4, 3, 1, 1), _ConvBnRelu(C * 4, C * 4, 3, 1, 1)]
        self.enc = nn.Sequential(*layers)
        self.out_channels = C * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class LeRes(nn.Module):
    """LeRes: ResNet encoder + channel-attention decoder for monocular depth."""

    def __init__(self, C: int = 16, deep: bool = False) -> None:
        super().__init__()
        self.enc = _LeResEncoder(C, deep)
        fc = self.enc.out_channels
        self.attn = _ChannelAttention(fc)
        self.up1 = nn.Sequential(
            _ConvBnRelu(fc, fc // 2),
        )
        self.up2 = nn.Sequential(
            _ConvBnRelu(fc // 2, fc // 4),
        )
        self.depth_head = nn.Conv2d(fc // 4, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.enc(x)
        f = self.attn(f)
        d = _upsample(self.up1(f), (f.shape[2] * 2, f.shape[3] * 2))
        d = _upsample(self.up2(d), (d.shape[2] * 2, d.shape[3] * 2))
        d = self.depth_head(d)
        d = F.interpolate(d, x.shape[2:], mode="bilinear", align_corners=False)
        return d


# ══════════════════════════════════════════════════════════════
# PixelFormer: Swin encoder + SAM + adaptive bins decoder
# ══════════════════════════════════════════════════════════════


class _WindowAttn(nn.Module):
    """Compact windowed multi-head self-attention (Swin-like, no shift)."""

    def __init__(self, dim: int, n_heads: int = 2, window_size: int = 4) -> None:
        super().__init__()
        self.ws = window_size
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        B, C, H, W = x.shape
        ws = min(self.ws, H, W)
        # partition into windows
        nwh, nww = H // ws, W // ws
        if nwh == 0 or nww == 0:
            # too small for windowing, do global attention
            tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            n = self.norm(tokens)
            a, _ = self.attn(n, n, n)
            tokens = tokens + a
            tokens = tokens + self.ff(tokens)
            return tokens.transpose(1, 2).view(B, C, H, W)
        # window partition
        x_pad = x[:, :, : nwh * ws, : nww * ws]
        x_win = x_pad.view(B, C, nwh, ws, nww, ws)
        x_win = x_win.permute(0, 2, 4, 3, 5, 1).contiguous()
        x_win = x_win.view(B * nwh * nww, ws * ws, C)
        n = self.norm(x_win)
        a, _ = self.attn(n, n, n)
        x_win = x_win + a
        x_win = x_win + self.ff(x_win)
        # merge windows
        x_win = x_win.view(B, nwh, nww, ws, ws, C)
        x_win = x_win.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = x_win.view(B, C, nwh * ws, nww * ws)
        if H > nwh * ws or W > nww * ws:
            out = F.interpolate(out, (H, W), mode="nearest")
        return out


class _SwinLiteEncoder(nn.Module):
    """Compact Swin-like encoder for PixelFormer."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, C, 4, 4),
            _WindowAttn(C),
        )
        self.merge1 = nn.Sequential(
            nn.Conv2d(C, C * 2, 2, 2),
            _WindowAttn(C * 2),
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 2, 2),
            _WindowAttn(C * 4),
        )
        self.merge3 = nn.Sequential(
            nn.Conv2d(C * 4, C * 8, 2, 2),
            _WindowAttn(C * 8),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.merge1(f1)
        f3 = self.merge2(f2)
        f4 = self.merge3(f3)
        return [f1, f2, f3, f4]


class _SAM(nn.Module):
    """Skip Attention Module (SAM): cross-attention between decoder and skip features."""

    def __init__(self, skip_c: int, dec_c: int, out_c: int) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(dec_c, out_c, 1)
        self.k_proj = nn.Conv2d(skip_c, out_c, 1)
        self.v_proj = nn.Conv2d(skip_c, out_c, 1)
        self.out_proj = nn.Conv2d(out_c, out_c, 1)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        # align decoder spatial size to skip
        if dec.shape[2:] != skip.shape[2:]:
            dec = F.interpolate(dec, skip.shape[2:], mode="bilinear", align_corners=False)
        B, C, H, W = skip.shape
        q = self.q_proj(dec).flatten(2).transpose(1, 2)  # (B, HW, C_out)
        k = self.k_proj(skip).flatten(2).transpose(1, 2)
        v = self.v_proj(skip).flatten(2).transpose(1, 2)
        scale = q.shape[-1] ** -0.5
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)  # (B, HW, C_out)
        out = out.transpose(1, 2).view(B, -1, H, W)
        return self.out_proj(out)


class _AdaptiveBinsHead(nn.Module):
    """Bin-center predicting depth head (simplified adaptive bins)."""

    def __init__(self, in_c: int, n_bins: int = 8) -> None:
        super().__init__()
        self.bin_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_c, n_bins),
            nn.Softmax(dim=1),
        )
        self.range_proj = nn.Conv2d(in_c, n_bins, 1)
        self.depth_proj = nn.Conv2d(n_bins, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # per-pixel bin probabilities
        bin_w = self.range_proj(x)  # (B, n_bins, H, W)
        bin_w = F.softmax(bin_w, dim=1)
        return self.depth_proj(bin_w)


class PixelFormer(nn.Module):
    """PixelFormer: Swin encoder + SAM + adaptive bins head."""

    def __init__(self, C: int = 16, n_bins: int = 8) -> None:
        super().__init__()
        self.enc = _SwinLiteEncoder(C)
        self.sam3 = _SAM(C * 4, C * 8, C * 4)
        self.sam2 = _SAM(C * 2, C * 4, C * 2)
        self.sam1 = _SAM(C, C * 2, C)
        self.depth_head = _AdaptiveBinsHead(C, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.enc(x)
        d = self.sam3(f3, f4)
        d = self.sam2(f2, d)
        d = self.sam1(f1, d)
        d = _upsample(self.depth_head(d), x.shape[2:])
        return d


# ──────────────────────────────────────────────────────────────
# Builders + entries
# ──────────────────────────────────────────────────────────────


def build_fcrn_resnet50_depth() -> nn.Module:
    """FCRN: ResNet encoder + Up-Projection decoder, compact (C=16)."""
    return FCRN(C=16)


def build_glpdepth_swin() -> nn.Module:
    """GLPDepth: hierarchical encoder + Selective Feature Fusion, compact."""
    return GLPDepth(C=16)


def build_lapdepth_resnet() -> nn.Module:
    """LapDepth: ResNet encoder + Laplacian-pyramid decoder, compact."""
    return LapDepth(C=16, n_levels=3)


def build_leres() -> nn.Module:
    """LeRes: ResNet encoder + attention decoder for mono depth, compact."""
    return LeRes(C=16, deep=False)


def build_leres_resnext101() -> nn.Module:
    """LeRes ResNeXt101: deeper encoder variant of LeRes, compact."""
    return LeRes(C=16, deep=True)


def build_pixelformer() -> nn.Module:
    """PixelFormer: Swin+SAM + adaptive-bins depth head, compact."""
    return PixelFormer(C=16, n_bins=8)


def example_input() -> torch.Tensor:
    """RGB image (1, 3, 64, 128)."""
    return torch.randn(1, 3, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "FCRN (Up-Projection ResNet monocular depth)",
        "build_fcrn_resnet50_depth",
        "example_input",
        "2016",
        "DC",
    ),
    (
        "GLPDepth (hierarchical encoder + SFF monocular depth)",
        "build_glpdepth_swin",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "LapDepth (Laplacian-pyramid decoder monocular depth)",
        "build_lapdepth_resnet",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "LeRes (ResNet + attention decoder monocular depth)",
        "build_leres",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "LeRes ResNeXt101 (deeper encoder monocular depth variant)",
        "build_leres_resnext101",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "PixelFormer (Swin + Skip Attention Module + adaptive bins depth)",
        "build_pixelformer",
        "example_input",
        "2022",
        "DC",
    ),
]
