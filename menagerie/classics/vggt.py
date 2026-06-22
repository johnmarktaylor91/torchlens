"""VGGT: Visual Geometry Grounded Transformer.

Wang et al., CVPR 2025.
Paper: https://arxiv.org/abs/2503.11651
Source: https://github.com/facebookresearch/vggt

VGGT is a feed-forward transformer that, from one or many views of a scene,
directly predicts camera parameters, depth maps, point maps, and 3D point
tracks.  Its DISTINCTIVE mechanism is the ALTERNATING-ATTENTION backbone:

  - Each frame is patch-tokenized (DINO-style Conv2d patch-embed).
  - Per frame, one learned CAMERA token + a few learned REGISTER tokens are
    prepended to that frame's patch tokens.
  - The backbone is a stack of L alternating blocks: a FRAME-WISE
    self-attention layer (tokens attend only within their own frame) followed
    by a GLOBAL self-attention layer (all tokens of all frames attend jointly).
    This lets the model reason within and across views without an explicit
    geometry module.

Heads:
  - A small camera-pose MLP head reads the camera token of each frame.
  - A DPT-style convolutional head reads the patch tokens and upsamples to a
    dense per-frame depth map.

This faithful reimplementation captures the alternating frame/global attention
backbone and both heads at modest width (embed_dim=128, a few alternating
pairs).  Random init is the correct artifact for a structure atlas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHSA(nn.Module):
    """Plain multi-head self-attention over a (B, N, C) token sequence."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Block(nn.Module):
    """Pre-norm transformer block (attention + MLP)."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchEmbed(nn.Module):
    """DINO-style Conv2d patch tokenizer."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 128, patch: int = 8) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/p, W/p)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (H, W)  # (B, H*W, C)


class _DPTDepthHead(nn.Module):
    """DPT-style conv head: patch tokens -> dense depth map (1 channel)."""

    def __init__(self, embed_dim: int = 128, patch: int = 8) -> None:
        super().__init__()
        self.patch = patch
        self.conv1 = nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1)
        self.out = nn.Conv2d(embed_dim // 4, 1, 1)

    def forward(self, tokens: torch.Tensor, grid: tuple, out_hw: tuple) -> torch.Tensor:
        B, N, C = tokens.shape
        H, W = grid
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.conv2(x))
        x = self.out(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class VGGT(nn.Module):
    """VGGT alternating frame/global attention backbone + camera & depth heads."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 2,
        n_alt_pairs: int = 5,
        n_register: int = 4,
        patch: int = 8,
        in_ch: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_register = n_register
        self.patch = patch

        self.patch_embed = _PatchEmbed(in_ch, embed_dim, patch)
        # Special tokens: 1 camera token + n_register register tokens, learned per frame.
        self.camera_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_register, embed_dim))

        # Alternating blocks: each pair = (frame-wise block, global block).
        self.frame_blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(n_alt_pairs)]
        )
        self.global_blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(n_alt_pairs)]
        )

        # Camera-pose MLP head (off camera token): 9 = quaternion(4)+trans(3)+fov(2).
        self.camera_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 9),
        )
        self.depth_head = _DPTDepthHead(embed_dim, patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C, H, W) -- B=batch, S=number of frames.
        B, S, C, H, W = x.shape
        flat = x.reshape(B * S, C, H, W)
        patch_tokens, grid = self.patch_embed(flat)  # (B*S, P, embed)
        _P = patch_tokens.shape[1]

        cam = self.camera_token.expand(B * S, -1, -1)
        reg = self.register_tokens.expand(B * S, -1, -1)
        # Per-frame token sequence: [camera | registers | patches].
        tokens = torch.cat([cam, reg, patch_tokens], dim=1)  # (B*S, 1+R+P, embed)
        n_special = 1 + self.n_register
        T = tokens.shape[1]

        for fblk, gblk in zip(self.frame_blocks, self.global_blocks):
            # Frame-wise attention: each frame's tokens attend independently.
            tokens = fblk(tokens)  # already (B*S, T, embed) == per-frame batch
            # Global attention: all frames of a sample attend jointly.
            tokens = tokens.reshape(B, S * T, self.embed_dim)
            tokens = gblk(tokens)
            tokens = tokens.reshape(B * S, T, self.embed_dim)

        # Heads.
        cam_token = tokens[:, 0, :]  # (B*S, embed)
        _camera_pose = self.camera_head(cam_token)  # (B*S, 9) -- traced head, leaf

        patch_out = tokens[:, n_special:, :]  # (B*S, P, embed)
        depth = self.depth_head(patch_out, grid, (H, W))  # (B*S, 1, H, W)

        # Return the dense depth map, reshaped back to (B, S, 1, H, W).
        depth = depth.reshape(B, S, 1, H, W)
        return depth


def build_vggt() -> nn.Module:
    """Build VGGT (alternating frame/global attention, camera + depth heads)."""
    return VGGT(embed_dim=128, num_heads=2, n_alt_pairs=5, n_register=4, patch=8, in_ch=3)


def example_input() -> torch.Tensor:
    """Example multi-view tensor ``(1, 2, 3, 32, 32)`` = batch 1, 2 frames."""
    return torch.randn(1, 2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "VGGT (alternating frame/global attention geometry transformer)",
        "build_vggt",
        "example_input",
        "2025",
        "DC",
    ),
]
