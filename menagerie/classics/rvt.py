"""RVT: Robotic View Transformer for 3D object manipulation.

Goyal et al. (NVlabs), CoRL 2023.
Paper: https://arxiv.org/abs/2306.14896
Source: https://github.com/NVlabs/RVT

RVT re-renders a point cloud into a small set of orthographic VIRTUAL VIEWS,
then runs a multi-view transformer that jointly reasons across views to predict
the robot end-effector pose.  Its DISTINCTIVE mechanism:

  - Per-view Conv patch-embed turns each virtual view into patch tokens.
  - A multi-view transformer applies CROSS-VIEW attention: tokens from ALL V
    views are concatenated and attend jointly, so the network fuses evidence
    across the re-rendered virtual cameras.
  - Per-view dense conv heads decode per-view translation HEATMAPS, and a global
    MLP head predicts rotation / gripper / collision actions.

RENDER-FREE FAITHFUL CORE: the original point-cloud -> virtual-view rendering
step needs PyTorch3D, which is out of scope.  We therefore take the ALREADY-
RENDERED virtual views as input (V views, each ~7 channels: rgb + depth +
coords) and faithfully reimplement the transformer core + heads.  The
architecture captured here is exactly RVT's cross-view multi-view transformer;
only the upstream renderer (a fixed projection, not a learned module) is
provided as input.

Modest width (embed_dim=128); forward() returns the per-view heatmap stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAttention(nn.Module):
    """Multi-head self-attention over the concatenated multi-view token set."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RVT(nn.Module):
    """Robotic View Transformer core: cross-view transformer + per-view heads."""

    def __init__(
        self,
        n_views: int = 5,
        in_ch: int = 7,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        patch: int = 4,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.embed_dim = embed_dim
        self.patch = patch

        # Per-view Conv patch-embed (shared across views).
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        # Learned per-view embedding so the transformer knows which view a token came from.
        self.view_embed = nn.Parameter(torch.zeros(1, n_views, 1, embed_dim))

        # Multi-view transformer (cross-view attention over all V views' tokens).
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Per-view dense conv head -> translation heatmap (1 channel per view).
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, 1),
        )
        # Global MLP head -> rotation (quat 4) + gripper (1) + collision (1).
        self.global_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 6),
        )

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        # views: (B, V, in_ch, H, W) already-rendered virtual views.
        B, V, Cin, H, W = views.shape
        flat = views.reshape(B * V, Cin, H, W)
        feat = self.patch_embed(flat)  # (B*V, embed, gh, gw)
        gh, gw = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2)  # (B*V, P, embed)
        P = tokens.shape[1]
        tokens = tokens.reshape(B, V, P, self.embed_dim)
        tokens = tokens + self.view_embed  # add per-view embedding

        # Concatenate all views' tokens -> cross-view attention.
        cv = tokens.reshape(B, V * P, self.embed_dim)
        for blk in self.blocks:
            cv = blk(cv)
        cv = self.norm(cv)

        # Per-view heatmaps.
        per_view = cv.reshape(B, V, P, self.embed_dim)
        pv_flat = per_view.reshape(B * V, P, self.embed_dim).transpose(1, 2)
        pv_map = pv_flat.reshape(B * V, self.embed_dim, gh, gw)
        heat = self.heatmap_head(pv_map)  # (B*V, 1, gh, gw)
        heat = F.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)
        heat = heat.reshape(B, V, 1, H, W)

        # Global action head reads the mean-pooled multi-view context.
        global_ctx = cv.mean(dim=1)  # (B, embed)
        _action = self.global_head(global_ctx)  # (B, 6) -- computed for fidelity

        # Return the per-view heatmap stack (the translation prediction).
        return heat


def build_rvt() -> nn.Module:
    """Build RVT cross-view transformer core (render-free; views as input)."""
    return RVT(n_views=5, in_ch=7, embed_dim=128, depth=4, num_heads=4, patch=4)


def example_input() -> torch.Tensor:
    """Example pre-rendered virtual views ``(1, 5, 7, 32, 32)`` = V=5 views, 7ch."""
    return torch.randn(1, 5, 7, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "RVT (multi-view cross-view robotic view transformer)",
        "build_rvt",
        "example_input",
        "2023",
        "DC",
    ),
]
