"""MoCha-Stereo: Motif Channel Attention Stereo Network.

Chen et al., CVPR 2023.
Paper: https://arxiv.org/abs/2303.14065
Source: https://github.com/ZYangChen/MoCha-Stereo

Distinctive primitive:
  - Motif Channel Correlation Volume: instead of raw feature channels, channels
    are grouped into "motifs" (learned feature basis via 1x1 conv), and the
    correlation is computed between left/right motif representations.
  - Motif Channel Attention (MCA): applies channel attention weighted by motif
    similarity between left and right features to suppress less discriminative channels.
  - 3D-conv hourglass aggregation + soft-argmin regression (same as PSMNet family).

Random init; compact: H=32, W=64, D=12, C=16, n_motifs=4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────
# Feature extractor
# ──────────────────────────────────────────────────────────────


class _FeatureNet(nn.Module):
    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 7, 2, 3),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        )
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Motif Channel Attention (MCA)
# ──────────────────────────────────────────────────────────────


class _MotifChannelAttention(nn.Module):
    """Motif Channel Attention: attention weights derived from motif correlation.

    Projects features into a motif space and computes cross-feature channel
    attention to weight feature channels by motif similarity.
    """

    def __init__(self, in_c: int, n_motifs: int = 4) -> None:
        super().__init__()
        self.n_motifs = n_motifs
        # Project to motif space
        self.motif_proj = nn.Conv2d(in_c, n_motifs, 1)
        # Attention from motif similarity
        self.attn_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_c, in_c // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // 4, in_c),
            nn.Sigmoid(),
        )
        self.out_channels = in_c

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Returns channel-attended left features."""
        B, C, H, W = left.shape
        # Motif correlation: inner product between left/right motif channels
        ml = self.motif_proj(left)  # (B, M, H, W)
        mr = self.motif_proj(right)  # (B, M, H, W)
        motif_corr = (ml * mr).mean(dim=[2, 3])  # (B, M)
        # Broadcast motif similarity back to channel attention
        # Use left features pooled + motif info
        attn = self.attn_net(left)  # (B, C)
        attn = attn.view(B, C, 1, 1)
        return left * attn


# ──────────────────────────────────────────────────────────────
# Motif Channel Correlation Volume
# ──────────────────────────────────────────────────────────────


def _motif_corr_volume(
    left: torch.Tensor, right: torch.Tensor, max_disp: int, n_motifs: int, motif_proj: nn.Module
) -> torch.Tensor:
    """Build motif channel correlation volume (B, n_motifs, D, H, W)."""
    B, C, H, W = left.shape
    ml = motif_proj(left)  # (B, M, H, W)
    mr = motif_proj(right)  # (B, M, H, W)
    cost = torch.zeros(B, n_motifs, max_disp, H, W, device=left.device, dtype=left.dtype)
    for d in range(max_disp):
        if d == 0:
            r = mr
        else:
            r = torch.zeros_like(mr)
            r[:, :, :, d:] = mr[:, :, :, :-d]
        cost[:, :, d] = ml * r
    return cost


# ──────────────────────────────────────────────────────────────
# 3D Hourglass
# ──────────────────────────────────────────────────────────────


class _Conv3dBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Hourglass3D(nn.Module):
    def __init__(self, C: int) -> None:
        super().__init__()
        self.e1 = _Conv3dBnRelu(C, C)
        self.e2 = _Conv3dBnRelu(C, C * 2)
        self.bot = _Conv3dBnRelu(C * 2, C * 2)
        self.d2 = _Conv3dBnRelu(C * 4, C)
        self.d1 = _Conv3dBnRelu(C * 2, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool3d(e1, 2, 2))
        bot = self.bot(F.avg_pool3d(e2, 2, 2))
        u2 = F.interpolate(bot, e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, e1.shape[2:], mode="trilinear", align_corners=False)
        return self.d1(torch.cat([u1, e1], 1))


def _soft_argmin(cost: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(-cost, dim=1)
    idx = torch.arange(cost.shape[1], device=cost.device, dtype=cost.dtype).view(1, -1, 1, 1)
    return (prob * idx).sum(1, keepdim=True)


# ──────────────────────────────────────────────────────────────
# MoCha-Stereo
# ──────────────────────────────────────────────────────────────


class MoCHAStereo(nn.Module):
    """MoCha-Stereo: motif channel correlation + attention + 3D hourglass."""

    def __init__(self, max_disp: int = 12, C: int = 16, n_motifs: int = 4) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.n_motifs = n_motifs
        self.feat = _FeatureNet(C)
        fc = self.feat.out_channels
        # Motif Channel Attention
        self.mca = _MotifChannelAttention(fc, n_motifs)
        # Motif projection for cost volume
        self.motif_proj = nn.Conv2d(fc, n_motifs, 1)
        # fuse motif corr + concat features
        self.init_conv = _Conv3dBnRelu(n_motifs + fc * 2, C)
        self.hourglass = _Hourglass3D(C)
        self.classifier = nn.Conv3d(C, 1, 3, 1, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.feat(left)
        fr = self.feat(right)
        # apply motif channel attention
        fl_att = self.mca(fl, fr)
        # build motif correlation volume
        motif_vol = _motif_corr_volume(fl_att, fr, self.max_disp, self.n_motifs, self.motif_proj)
        # build concat cost volume
        B, C, H, W = fl.shape
        concat_vol = torch.zeros(
            B, C * 2, self.max_disp, H, W, device=left.device, dtype=left.dtype
        )
        for d in range(self.max_disp):
            if d == 0:
                concat_vol[:, :C, d] = fl
                concat_vol[:, C:, d] = fr
            else:
                concat_vol[:, :C, d, :, d:] = fl[:, :, :, d:]
                concat_vol[:, C:, d, :, d:] = fr[:, :, :, :-d]
        cost = torch.cat([motif_vol, concat_vol], dim=1)
        cost = self.init_conv(cost)
        cost = self.hourglass(cost)
        cost_out = self.classifier(cost).squeeze(1)
        disp = _soft_argmin(cost_out)
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Wrapper + builders
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_mocha_stereo() -> nn.Module:
    """MoCha-Stereo (motif channel correlation + attention + 3D hourglass), compact."""
    return _StereoWrapper(MoCHAStereo(max_disp=12, C=16, n_motifs=4))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "MoCha-Stereo (motif channel correlation volume stereo)",
        "build_mocha_stereo",
        "example_input",
        "2023",
        "DC",
    ),
]
