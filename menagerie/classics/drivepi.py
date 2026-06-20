"""DrivePI: spatial-aware 4D multimodal LLM for autonomous driving.

Liu et al., CVPR 2026.
Paper: https://arxiv.org/abs/2512.12799
Source: https://github.com/happinesslz/DrivePI

DrivePI is a unified driving model: a language-model decoder backbone ingests a
packed sequence of [image-feature tokens | BEV / point tokens | language tokens]
and drives THREE parallel task heads off the shared backbone.  Its DISTINCTIVE
mechanism:

  - A tiny conv image encoder turns multi-view images into image-feature tokens.
  - These are packed with BEV/point tokens and language token embeddings into a
    single sequence consumed by a SMALL Qwen2.5-style transformer decoder
    backbone (causal attention, RMSNorm-style pre-norm).
  - THREE heads run in parallel on the backbone output:
      * a 3D-OCCUPANCY head (conv/MLP -> voxel occupancy logits),
      * an occupancy-FLOW head (MLP -> per-voxel flow), and
      * a PLANNING / action head (MLP -> future trajectory waypoints).
  - Unified 4D (3D space + time) scene understanding and planning in one model.

This faithful reimplementation captures the shared LM backbone with the three
parallel occupancy / flow / planning heads at modest width (embed_dim=128,
4 blocks, vocab=256).  The wrapper builds the language token ids internally so
forward takes a single image tensor.  forward() returns the planning waypoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention (Qwen-style decoder attention)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class _DecoderBlock(nn.Module):
    """Qwen2.5-style pre-norm decoder block with SwiGLU-ish MLP."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        h = self.down(self.act(self.gate(h)) * self.up(h))
        return x + h


class _ImageEncoder(nn.Module):
    """Tiny conv encoder: multi-view image -> image-feature tokens."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 128, n_tokens: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> 16 tokens
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        f = self.pool(f)  # (B, embed, 4, 4)
        return f.flatten(2).transpose(1, 2)  # (B, 16, embed)


class DrivePI(nn.Module):
    """DrivePI: shared LM backbone with occupancy / flow / planning heads."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        n_lang: int = 8,
        n_bev: int = 8,
        voxel_classes: int = 8,
        n_waypoints: int = 6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_lang = n_lang
        self.n_bev = n_bev

        self.image_encoder = _ImageEncoder(3, embed_dim)
        # BEV / point tokens (learned, stand-in for an upstream BEV encoder).
        self.bev_tokens = nn.Parameter(torch.zeros(1, n_bev, embed_dim))
        self.lang_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Shared LM-decoder backbone.
        self.blocks = nn.ModuleList([_DecoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Three parallel heads.
        self.occupancy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, voxel_classes),
        )
        self.flow_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),  # per-voxel 3D flow
        )
        self.planning_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_waypoints * 2),  # (x, y) per waypoint
        )
        self.n_waypoints = n_waypoints

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        img_tok = self.image_encoder(image)  # (B, 16, embed)
        bev_tok = self.bev_tokens.expand(B, -1, -1)  # (B, n_bev, embed)
        lang_ids = torch.arange(self.n_lang, device=image.device).unsqueeze(0)
        lang_ids = (lang_ids.expand(B, -1)) % self.vocab_size
        lang_tok = self.lang_embed(lang_ids)  # (B, n_lang, embed)

        # Packed sequence: [image | BEV/point | language].
        x = torch.cat([img_tok, bev_tok, lang_tok], dim=1)
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        n_img = img_tok.shape[1]
        img_hidden = x[:, :n_img, :]
        bev_hidden = x[:, n_img : n_img + self.n_bev, :]
        lang_hidden = x[:, n_img + self.n_bev :, :]

        # Run all three heads (parallel, off the shared backbone).
        _occ = self.occupancy_head(img_hidden)  # (B, n_img, voxel_classes)
        _flow = self.flow_head(bev_hidden)  # (B, n_bev, 3)
        plan_ctx = lang_hidden.mean(dim=1)  # (B, embed)
        waypoints = self.planning_head(plan_ctx).reshape(B, self.n_waypoints, 2)

        # Return planning trajectory waypoints.
        return waypoints


def build_drivepi() -> nn.Module:
    """Build DrivePI (shared LM backbone + occupancy / flow / planning heads)."""
    return DrivePI(
        vocab_size=256,
        embed_dim=128,
        depth=4,
        num_heads=4,
        n_lang=8,
        n_bev=8,
        voxel_classes=8,
        n_waypoints=6,
    )


def example_input() -> torch.Tensor:
    """Example multi-view image ``(1, 3, 64, 64)`` (BEV/language tokens internal)."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "DrivePI (unified 4D MLLM, occupancy / flow / planning heads)",
        "build_drivepi",
        "example_input",
        "2026",
        "DC",
    ),
]
