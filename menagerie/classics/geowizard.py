"""GeoWizard: diffusion-based joint depth and surface-normal estimation.

Fu et al. (2024), "GeoWizard: Unleashing the Diffusion Priors for 3D Geometry
Estimation from a Single Image".  ECCV 2024.  arXiv:2403.12013.
Source: https://github.com/fuxiao0719/GeoWizard

Distinctive primitives:
  1. STABLE DIFFUSION U-NET backbone: convolutional U-Net with residual + attention
     blocks, timestep conditioning via AdaGN, cross-attention to text/geometry tokens.
  2. JOINT DEPTH + NORMAL OUTPUT: the denoiser produces TWO output heads (depth, normal)
     sharing the U-Net backbone.
  3. GEOMETRY SWITCHER / DOMAIN-SWITCH EMBEDDING: a learned embedding distinguishes
     the "geometry mode" (depth vs normals vs joint), injected into the timestep
     embedding to steer the shared backbone to each domain.

This compact reimplementation reproduces:
  - A small SD-style UNet block: ResBlock (with timestep AdaGN) + SpatialAttention.
  - A domain-switch embedding (2 modes: depth, normal).
  - Two output conv heads projecting to depth (1 ch) and normal (3 ch).
  - Wrapped to return (depth, normal) concatenated as a single (B, 4, H, W) tensor.

Compact config: C=32, 2 resblocks, spatial 8x8, time_emb_dim=32.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Sinusoidal timestep embedding
# ==============================================================


def get_timestep_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    """t: (B,) long or float -> (B, d) sinusoidal embedding."""
    half = d // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d)


# ==============================================================
# SD-style ResBlock with AdaGN timestep conditioning
# ==============================================================


class AdaGNResBlock(nn.Module):
    """Residual block with Adaptive Group Normalisation (AdaGN) for timestep + domain."""

    def __init__(self, c_in: int, c_out: int, t_emb_dim: int = 32, n_groups: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        # Timestep/domain scale + shift for norm2
        self.t_proj = nn.Linear(t_emb_dim, 2 * c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, c_in, H, W), t_emb: (B, t_emb_dim) -> (B, c_out, H, W)"""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # AdaGN: scale/shift from t_emb
        ts = self.t_proj(self.act(t_emb))  # (B, 2*c_out)
        scale, shift = ts.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ==============================================================
# Spatial Self-Attention
# ==============================================================


class SpatialSelfAttention(nn.Module):
    """1-head spatial self-attention over H*W tokens."""

    def __init__(self, c: int, n_groups: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(n_groups, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, C, HW)
        scale = C**-0.5
        attn = torch.softmax(Q.permute(0, 2, 1) @ K * scale, dim=-1)  # (B, HW, HW)
        out = (attn @ V.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        return x + self.out(out)


# ==============================================================
# Compact SD-style UNet block
# ==============================================================


class SDUNetBlock(nn.Module):
    """One encoder block: ResBlock + SpatialAttention."""

    def __init__(self, c_in: int, c_out: int, t_emb_dim: int = 32) -> None:
        super().__init__()
        self.res = AdaGNResBlock(c_in, c_out, t_emb_dim)
        self.attn = SpatialSelfAttention(c_out)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, t_emb))


# ==============================================================
# GeoWizard compact model
# ==============================================================


class GeoWizardDepthNormal(nn.Module):
    """Compact GeoWizard: SD-UNet + domain-switch embedding -> depth + normal heads.

    Input: (B, 3, H, W) noisy image + (B,) timestep + (B,) domain_id (0=depth, 1=normal).
    Wrapped into a single flat tensor input.
    Output: (B, 4, H, W) where channels 0 = depth, 1:4 = surface normal.
    """

    def __init__(self, c: int = 32, t_emb_dim: int = 32, n_blocks: int = 2) -> None:
        super().__init__()
        self.c = c
        self.t_emb_dim = t_emb_dim
        # Timestep embedding MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 2, t_emb_dim),
        )
        # Domain-switch embedding: 2 modes (depth / normal)
        self.domain_emb = nn.Embedding(2, t_emb_dim)
        # UNet encoder
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.enc_blocks = nn.ModuleList([SDUNetBlock(c, c, t_emb_dim) for _ in range(n_blocks)])
        # Middle
        self.mid = SDUNetBlock(c, c, t_emb_dim)
        # UNet decoder (skip connections)
        self.dec_blocks = nn.ModuleList([SDUNetBlock(2 * c, c, t_emb_dim) for _ in range(n_blocks)])
        # Output heads
        self.head_depth = nn.Conv2d(c, 1, 1)
        self.head_normal = nn.Conv2d(c, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3*H*W + 2) packed: flat image + [timestep_int, domain_id_int] as float.
        Output: (B, 4*H*W) packed depth+normal flat tensor.
        """
        B = x.shape[0]
        H = W = 8  # fixed compact spatial size
        img = x[:, : 3 * H * W].view(B, 3, H, W)
        meta = x[:, 3 * H * W :]  # (B, 2): [t_float, domain_float]
        t = meta[:, 0].long().clamp(0, 999)
        d = meta[:, 1].long().clamp(0, 1)

        t_emb = get_timestep_embedding(t, self.t_emb_dim)
        t_emb = self.t_mlp(t_emb)  # (B, t_emb_dim)
        d_emb = self.domain_emb(d)  # (B, t_emb_dim)
        cond = t_emb + d_emb  # (B, t_emb_dim)

        h = self.stem(img)  # (B, c, H, W)
        skips = []
        for blk in self.enc_blocks:
            h = blk(h, cond)
            skips.append(h)

        h = self.mid(h, cond)

        for blk in self.dec_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = blk(h, cond)

        depth = self.head_depth(h)  # (B, 1, H, W)
        normal = self.head_normal(h)  # (B, 3, H, W)
        out = torch.cat([depth, normal], dim=1)  # (B, 4, H, W)
        return out.reshape(B, -1)


def build_geowizard_depth_normal() -> nn.Module:
    return GeoWizardDepthNormal(c=32, t_emb_dim=32, n_blocks=2).eval()


def example_input() -> torch.Tensor:
    """(1, 3*8*8 + 2) = (1, 194) packed: flat image + [timestep=50, domain=0]."""
    x = torch.randn(1, 3 * 8 * 8 + 2)
    x[0, -2] = 50.0  # timestep
    x[0, -1] = 0.0  # domain=depth
    return x


MENAGERIE_ENTRIES = [
    (
        "GeoWizard depth-normal (SD-UNet + domain-switch embedding -> joint depth+normal heads)",
        "build_geowizard_depth_normal",
        "example_input",
        "2024",
        "DC",
    ),
]
