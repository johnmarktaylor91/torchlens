"""STTR: STereo TRansformer for depth estimation.

Li et al., ICCV 2021.
Paper: https://arxiv.org/abs/2011.02910
Source: https://github.com/mli0603/stereo-transformer

STTR treats stereo depth estimation as a sequence-to-sequence problem along
epipolar lines (horizontal scanlines). Given a left and right image, features
are extracted with a small CNN, then for each row the 1D token sequences from
left and right are processed with alternating self-attention (within each image)
and cross-attention (across images) Transformer layers. Relative positional
encoding (learned sinusoidal-like offsets) replaces absolute PE to handle
variable-length sequences. A final optimal-transport matching module (Sinkhorn)
produces a soft correspondence / disparity distribution; the disparity map is
the expected value under this distribution.

Compact faithfulness:
  - Small CNN feature extractor (3 layers, strided).
  - Positional encoding via learned linear on coordinate grids.
  - Alternating self/cross-attention along the width (epipolar) axis.
  - Disparity regression (soft-argmax over cost volume) replaces Sinkhorn to
    keep the graph tractable (Sinkhorn iterations would dominate the trace).
  - sttr: 2 attention blocks, dim=32; sttr_light: 1 block, dim=16.
  - Input: stacked (left, right) as (1, 6, H, W) — left=[:3], right=[3:].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small CNN feature extractor
# ---------------------------------------------------------------------------


class FeatureExtractor(nn.Module):
    """Lightweight 3-layer CNN: (B, 3, H, W) -> (B, C, H, W/4)."""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Relative positional encoding
# ---------------------------------------------------------------------------


class RelativePositionalEncoding(nn.Module):
    """Learnable relative PE: maps position offsets to bias added to attention logits."""

    def __init__(self, max_len: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        # Relative offset table: [-max_len .. max_len]
        self.max_len = max_len
        self.rel_bias = nn.Embedding(2 * max_len + 1, num_heads)

    def forward(self, L: int) -> torch.Tensor:
        """Return (L, L, num_heads) relative bias."""
        positions = torch.arange(L, device=self.rel_bias.weight.device)
        offsets = positions.unsqueeze(1) - positions.unsqueeze(0)  # (L, L)
        offsets = offsets.clamp(-self.max_len, self.max_len) + self.max_len
        bias = self.rel_bias(offsets)  # (L, L, heads)
        return bias


# ---------------------------------------------------------------------------
# Self-attention and cross-attention along the epipolar axis
# ---------------------------------------------------------------------------


class EpipolarSelfAttn(nn.Module):
    """Multi-head self-attention over a 1D token sequence (epipolar scanline)."""

    def __init__(self, dim: int, num_heads: int = 4, max_len: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pe = RelativePositionalEncoding(max_len, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*H, W, C)
        BH, L, C = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(BH, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BH, heads, L, L)
        # Add relative positional bias
        rel_bias = self.rel_pe(L).permute(2, 0, 1)  # (heads, L, L)
        attn = attn + rel_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(BH, L, C)
        x = self.proj(x)
        return residual + x


class EpipolarCrossAttn(nn.Module):
    """Multi-head cross-attention: query from left, key/value from right (epipolar)."""

    def __init__(self, dim: int, num_heads: int = 4, max_len: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.rel_pe = RelativePositionalEncoding(max_len, num_heads)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        # q_in, kv_in: (B*H, W, C)
        BH, L, C = q_in.shape
        residual = q_in
        q_in_n = self.norm_q(q_in)
        kv_in_n = self.norm_kv(kv_in)
        q = self.q(q_in_n).reshape(BH, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv_in_n).reshape(BH, L, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.rel_pe(L).permute(2, 0, 1)
        attn = attn + rel_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(BH, L, C)
        x = self.proj(x)
        return residual + x


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# STTR Transformer block: self + cross + FFN
# ---------------------------------------------------------------------------


class STTRBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, max_len: int = 64) -> None:
        super().__init__()
        self.self_attn_l = EpipolarSelfAttn(dim, num_heads, max_len)
        self.self_attn_r = EpipolarSelfAttn(dim, num_heads, max_len)
        self.cross_attn_l = EpipolarCrossAttn(dim, num_heads, max_len)
        self.cross_attn_r = EpipolarCrossAttn(dim, num_heads, max_len)
        self.ffn_l = FFN(dim, dim * 2)
        self.ffn_r = FFN(dim, dim * 2)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention within each image
        left = self.self_attn_l(left)
        right = self.self_attn_r(right)
        # Cross-attention across images
        left_new = self.cross_attn_l(left, right)
        right_new = self.cross_attn_r(right, left)
        left = self.ffn_l(left_new)
        right = self.ffn_r(right_new)
        return left, right


# ---------------------------------------------------------------------------
# Disparity regression head
# ---------------------------------------------------------------------------


class DisparityRegressionHead(nn.Module):
    """Soft-argmax disparity regression from cross-correlation."""

    def __init__(self, dim: int, max_disp: int = 16) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.proj = nn.Linear(dim, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # left, right: (B*H, W, C)
        BH, _, C = left.shape
        B = BH // H

        # Build 1D cost volume by correlation at each disparity shift
        max_d = min(self.max_disp, W)
        costs = []
        for d in range(max_d):
            if d == 0:
                cost = (left * right).sum(-1, keepdim=True)  # (BH, W, 1)
            else:
                shifted_right = F.pad(right[:, d:, :], (0, 0, 0, d))
                cost = (left * shifted_right).sum(-1, keepdim=True)
            costs.append(cost)
        cost_vol = torch.cat(costs, dim=-1)  # (BH, W, max_d)
        cost_vol = cost_vol.softmax(dim=-1)
        disp_vals = torch.arange(max_d, dtype=left.dtype, device=left.device)
        disparity = (cost_vol * disp_vals).sum(-1)  # (BH, W)

        # Upsample back to original resolution (approx x4)
        disparity = disparity.view(B, H, W)
        disparity = F.interpolate(
            disparity.unsqueeze(1), scale_factor=4.0, mode="bilinear", align_corners=False
        ).squeeze(1)
        return disparity


# ---------------------------------------------------------------------------
# Full STTR model
# ---------------------------------------------------------------------------


class STTR(nn.Module):
    """STereo TRansformer (compact reimplementation, random init).

    Input: (B, 6, H, W) where [:3] = left, [3:] = right.
    Output: (B, H, W) disparity map.
    """

    def __init__(self, dim: int = 32, num_blocks: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.feat_left = FeatureExtractor(dim)
        self.feat_right = FeatureExtractor(dim)
        self.blocks = nn.ModuleList(
            [STTRBlock(dim, num_heads, max_len=64) for _ in range(num_blocks)]
        )
        self.disp_head = DisparityRegressionHead(dim, max_disp=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, H, W)
        left_img = x[:, :3]
        right_img = x[:, 3:]
        B, _, H, W = left_img.shape

        # Feature extraction: (B, C, H', W') with H'=H/4, W'=W/4
        fl = self.feat_left(left_img)  # (B, C, H4, W4)
        fr = self.feat_right(right_img)

        _, C, H4, W4 = fl.shape

        # Reshape to (B*H4, W4, C) for epipolar processing
        fl = fl.permute(0, 2, 3, 1).reshape(B * H4, W4, C)
        fr = fr.permute(0, 2, 3, 1).reshape(B * H4, W4, C)

        # Alternating self/cross attention blocks
        for blk in self.blocks:
            fl, fr = blk(fl, fr)

        # Disparity regression
        disp = self.disp_head(fl, fr, H4, W4)  # (B, H, W) approx
        return disp


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_sttr() -> nn.Module:
    """Build STTR (STereo TRansformer, 2 blocks, dim=32)."""
    return STTR(dim=32, num_blocks=2, num_heads=4)


def build_sttr_light() -> nn.Module:
    """Build STTR-Light (1 block, dim=16, lighter variant)."""
    return STTR(dim=16, num_blocks=1, num_heads=2)


def example_input() -> torch.Tensor:
    """Stacked left+right image pair: (1, 6, 64, 64)."""
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "STTR (STereo TRansformer, epipolar self/cross attention + disparity regression)",
        "build_sttr",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "STTR-Light (lightweight STereo TRansformer, 1 block)",
        "build_sttr_light",
        "example_input",
        "2021",
        "DC",
    ),
]
