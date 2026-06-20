"""DeltaWorld / DeltaTok: "A Frame is Worth One Token".

Amazon FAR, 2026.
Paper: https://arxiv.org/abs/2604.04913
Source: https://github.com/amazon-far/deltatok

DeltaTok / DeltaWorld is a video world model built on an extreme temporal
tokenizer.  Its DISTINCTIVE mechanism:

  - DeltaTok encoder: each frame is first passed through a small (frozen) vision
    feature extractor (a VFM stand-in: a tiny conv).  The FEATURE DIFFERENCE
    between consecutive frames is computed, and each difference is encoded by a
    pooling MLP into ONE continuous "delta" token.  A clip of T frames thus
    collapses to a 1D sequence of (T-1) delta tokens -- one token per frame
    transition.
  - DeltaWorld: a small CAUSAL Transformer runs over the 1D delta-token sequence
    and autoregressively predicts the NEXT delta token (future prediction),
    i.e. it world-models in the compressed one-token-per-frame space.

This faithful reimplementation captures the collapse-a-video-to-one-token-per-
frame-delta step and the autoregressive world model over those tokens at modest
width.  forward() returns the predicted next delta token (and the delta
sequence is the latent).  Random init is the correct artifact.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _FrameFeatureExtractor(nn.Module):
    """Tiny conv VFM stand-in: per-frame feature map -> pooled feature vector."""

    def __init__(self, in_ch: int = 3, feat_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_ch = feat_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        return self.pool(f).flatten(1)  # (B, feat_ch)


class _CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention over the delta-token sequence."""

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


class _CausalBlock(nn.Module):
    """Pre-norm causal transformer block."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DeltaWorld(nn.Module):
    """DeltaTok one-token-per-frame-delta encoder + DeltaWorld AR world model."""

    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 64,
        token_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.feature_extractor = _FrameFeatureExtractor(in_ch, feat_ch)
        # Encode each consecutive-frame feature DIFFERENCE into ONE delta token.
        self.delta_encoder = nn.Sequential(
            nn.Linear(feat_ch, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.token_dim = token_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, token_dim))
        # DeltaWorld AR transformer over the delta-token sequence.
        self.blocks = nn.ModuleList([_CausalBlock(token_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(token_dim)
        # Predict the next delta token.
        self.predict_next = nn.Linear(token_dim, token_dim)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, T, C, H, W) short video.
        B, T, C, H, W = clip.shape
        flat = clip.reshape(B * T, C, H, W)
        feats = self.feature_extractor(flat).reshape(B, T, -1)  # (B, T, feat_ch)

        # Feature difference between consecutive frames -> (T-1) deltas.
        diffs = feats[:, 1:, :] - feats[:, :-1, :]  # (B, T-1, feat_ch)
        delta_tokens = self.delta_encoder(diffs)  # (B, T-1, token_dim)
        L = delta_tokens.shape[1]
        delta_tokens = delta_tokens + self.pos_embed[:, :L, :]

        h = delta_tokens
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)

        # AR next-delta prediction: take last position -> predicted next token.
        next_delta = self.predict_next(h[:, -1, :])  # (B, token_dim)
        return next_delta


def build_deltaworld() -> nn.Module:
    """Build DeltaWorld (one-token-per-frame-delta encoder + AR world model)."""
    return DeltaWorld(in_ch=3, feat_ch=64, token_dim=128, depth=4, num_heads=4)


def example_input() -> torch.Tensor:
    """Example short clip ``(1, 4, 3, 32, 32)`` = batch 1, 4 frames."""
    return torch.randn(1, 4, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "DeltaWorld / DeltaTok (one-token-per-frame-delta video world model)",
        "build_deltaworld",
        "example_input",
        "2026",
        "DC",
    ),
]
