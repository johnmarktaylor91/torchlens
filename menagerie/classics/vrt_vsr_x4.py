"""VRT: Video Restoration Transformer (VSR x4).

Liang et al., IEEE TIP 2024 (arXiv 2201.12288).
Paper: https://arxiv.org/abs/2201.12288
Source: https://github.com/JingyunLiang/VRT

Distinctive primitive: Temporal Mutual Self-Attention (TMSA).
VRT processes a short video clip using Temporal Mutual Self-Attention blocks
that perform window attention JOINTLY across the temporal and spatial dimensions.
Each TMSA block:
  1. Partitions the (T, H, W) feature volume into 3-D windows of shape
     (t_win, h_win, w_win).
  2. Performs multi-head self-attention over tokens in each window — this
     "mutual" attention lets each spatial position attend to its neighbourhood
     across adjacent frames.
  3. Followed by a standard MLP (FFN).
  4. Parallel warping: before or between TMSA stages, features from adjacent
     frames are aligned by optical flow warping.

After N TMSA blocks, pixel-shuffle x4 upsampling produces the SR output.

Compact: tiny clip (B=1, T=3 frames, C=3, H=32, W=32), small window and channel
widths, 2 TMSA blocks, 2 attention heads.
Input: (1, T, C, H, W) = (1, 3, 3, 32, 32) clip.
Output: (1, T, 3, H*4, W*4) x4 super-resolved clip.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp img (B,C,H,W) with flow (B,2,H,W)."""
    B, C, H, W = img.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=flow.dtype, device=flow.device),
        torch.arange(W, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=0).unsqueeze(0) + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(
        img, grid.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True
    )


# -----------------------------------------------------------------------
# TMSA: Temporal Mutual Self-Attention block
# -----------------------------------------------------------------------


class TMSABlock(nn.Module):
    """Temporal Mutual Self-Attention block over a (T,H,W) window.

    For the compact atlas we use a single global window (the whole feature
    volume is one window) to avoid padding complexity while faithfully showing
    the temporal multi-head attention primitive.
    """

    def __init__(self, dim: int, num_heads: int = 2, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Multi-head attention (self-attention over temporal-spatial tokens)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # MLP
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T*H*W, dim) flattened spatio-temporal tokens."""
        # Attention with residual
        shortcut = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = shortcut + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------
# Parallel warping module
# -----------------------------------------------------------------------


class ParallelWarp(nn.Module):
    """Warp adjacent frames to the reference using learned conv-estimated flow."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        # Small flow estimator: two frames -> 2ch flow
        self.flow_est = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 1),
        )

    def forward(self, ref: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        """Warp nbr toward ref. Both (B,ch,H,W)."""
        flow = self.flow_est(torch.cat([ref, nbr], dim=1))
        return backwarp(nbr, flow)


# -----------------------------------------------------------------------
# VRT model
# -----------------------------------------------------------------------


class VRTVSRx4(nn.Module):
    """Video Restoration Transformer x4 (compact TMSA + pixel-shuffle).

    Forward:
        x: (B, T, C_in, H, W) video clip.
    Returns:
        (B, T, 3, H*4, W*4) super-resolved clip.
    """

    def __init__(
        self, in_ch: int = 3, embed_dim: int = 32, n_tmsa: int = 2, num_heads: int = 2
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Shallow feature extraction per frame
        self.feat_extract = nn.Conv2d(in_ch, embed_dim, 3, padding=1)

        # Parallel warping between adjacent frames
        self.warp = ParallelWarp(embed_dim)

        # TMSA blocks
        self.tmsa_blocks = nn.ModuleList([TMSABlock(embed_dim, num_heads) for _ in range(n_tmsa)])

        # Reconstruction: conv + pixel-shuffle x4
        self.recon = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, 3 * 16, 3, padding=1),  # 16 = 4*4
        )
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Step 1: per-frame feature extraction
        x_feat = x.view(B * T, C, H, W)
        feats = self.feat_extract(x_feat)  # (B*T, embed_dim, H, W)
        feats = feats.view(B, T, self.embed_dim, H, W)

        # Step 2: parallel warping (warp each frame toward frame 0)
        ref = feats[:, 0]  # (B, embed_dim, H, W)
        warped = []
        for t in range(T):
            w = self.warp(ref, feats[:, t])
            warped.append(w)
        feats_aligned = torch.stack(warped, dim=1)  # (B, T, embed, H, W)

        # Step 3: TMSA — flatten to (B, T*H*W, dim)
        tokens = feats_aligned.view(B, T * H * W, self.embed_dim)
        for blk in self.tmsa_blocks:
            tokens = blk(tokens)
        feats_out = tokens.view(B, T, self.embed_dim, H, W)

        # Step 4: pixel-shuffle x4 per frame
        results = []
        for t in range(T):
            f = feats_out[:, t]  # (B, embed_dim, H, W)
            hr = self.pixel_shuffle(self.recon(f))  # (B, 3, H*4, W*4)
            results.append(hr)
        out = torch.stack(results, dim=1)  # (B, T, 3, H*4, W*4)
        return out


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_vrt_vsr_x4() -> nn.Module:
    return VRTVSRx4(in_ch=3, embed_dim=32, n_tmsa=2, num_heads=2)


def example_input_vrt() -> torch.Tensor:
    """Video clip (1, T=3, C=3, H=32, W=32)."""
    return torch.randn(1, 3, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "VRT-VSR x4 (Liang 2022, temporal mutual self-attention video SR)",
        "build_vrt_vsr_x4",
        "example_input_vrt",
        "2022",
        "DC",
    ),
]
