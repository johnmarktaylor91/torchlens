"""VibeToken: resolution-agnostic 1D image tokenizer.

Sony Research, CVPR 2026.
Paper: https://arxiv.org/abs/2604.24885

VibeToken is a 1D image tokenizer that compresses ANY image into a SHORT,
fixed-length sequence of K tokens, independent of the input resolution.  Its
DISTINCTIVE mechanism:

  - The image is patch-embedded (ViT-style Conv2d) into a variable number of
    patch tokens (depends on resolution).
  - A set of K LEARNED query tokens (fixed count, e.g. 32) cross-attend to the
    patch tokens through a few transformer blocks, COMPRESSING the image into a
    1D sequence of exactly K latent tokens regardless of resolution.
  - A quantization bottleneck (here a linear bottleneck + straight-through
    rounding) discretizes the K tokens.
  - A decoder reconstructs the image: learned patch queries (one per output
    patch) cross-attend back to the K latent tokens, then a small conv head
    un-patchifies to pixels.

This faithful reimplementation captures the fixed-length 1D token bottleneck and
the encode/quantize/decode cross-attention structure at modest width
(embed_dim=128, K=32).  forward() returns the reconstructed image.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CrossAttention(nn.Module):
    """Multi-head cross-attention: query tokens attend to key/value tokens."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q_in.shape
        Nk = kv_in.shape[1]
        q = self.q(q_in).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_in).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class _CrossBlock(nn.Module):
    """Pre-norm cross-attention + MLP block (queries refined against context)."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = _CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = q + self.cross(self.norm_q(q), self.norm_kv(kv))
        q = q + self.mlp(self.norm2(q))
        return q


class VibeToken(nn.Module):
    """VibeToken 1D image tokenizer: fixed K-token bottleneck encoder/decoder."""

    def __init__(
        self,
        in_ch: int = 3,
        embed_dim: int = 128,
        num_tokens: int = 32,
        patch: int = 4,
        n_enc_blocks: int = 3,
        n_dec_blocks: int = 3,
        code_dim: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.in_ch = in_ch

        # ViT patch-embed.
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        # K learned query tokens that compress the image into a 1D sequence.
        self.query_tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.enc_blocks = nn.ModuleList(
            [_CrossBlock(embed_dim, num_heads) for _ in range(n_enc_blocks)]
        )

        # Quantization bottleneck: linear down/up + straight-through rounding.
        self.to_code = nn.Linear(embed_dim, code_dim)
        self.from_code = nn.Linear(code_dim, embed_dim)

        # Decoder: learned per-output-patch queries cross-attend to K tokens.
        self.patch_pos = None  # set lazily from grid in forward
        self.dec_query = nn.Linear(embed_dim, embed_dim)
        self.dec_blocks = nn.ModuleList(
            [_CrossBlock(embed_dim, num_heads) for _ in range(n_dec_blocks)]
        )
        # Un-patchify head: each decoded patch token -> patch*patch*in_ch pixels.
        self.to_pixels = nn.Linear(embed_dim, patch * patch * in_ch)
        # Learned positional embeddings for output patch queries (lazy, capped).
        self.max_patches = 256
        self.out_pos = nn.Parameter(torch.zeros(1, self.max_patches, embed_dim))

    def _straight_through_round(self, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.round(x) - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.patch_embed(x)  # (B, embed, H/p, W/p)
        gh, gw = feat.shape[2], feat.shape[3]
        patch_tokens = feat.flatten(2).transpose(1, 2)  # (B, P, embed)
        P = patch_tokens.shape[1]

        # Encode: K queries compress patch tokens into a 1D length-K sequence.
        q = self.query_tokens.expand(B, -1, -1)
        for blk in self.enc_blocks:
            q = blk(q, patch_tokens)  # (B, K, embed)

        # Quantize.
        code = self.to_code(q)
        code = self._straight_through_round(code)
        latent = self.from_code(code)  # (B, K, embed) -- the 1D token bottleneck

        # Decode: P output-patch queries cross-attend to the K latent tokens.
        out_q = self.out_pos[:, :P, :].expand(B, -1, -1)
        out_q = self.dec_query(out_q)
        for blk in self.dec_blocks:
            out_q = blk(out_q, latent)  # (B, P, embed)

        # Un-patchify to image.
        pix = self.to_pixels(out_q)  # (B, P, patch*patch*in_ch)
        pix = pix.reshape(B, gh, gw, self.in_ch, self.patch, self.patch)
        pix = pix.permute(0, 3, 1, 4, 2, 5).reshape(B, self.in_ch, gh * self.patch, gw * self.patch)
        return pix


def build_vibetoken() -> nn.Module:
    """Build VibeToken (fixed K=32 1D image tokenizer, reconstruction output)."""
    return VibeToken(
        in_ch=3,
        embed_dim=128,
        num_tokens=32,
        patch=4,
        n_enc_blocks=3,
        n_dec_blocks=3,
        code_dim=32,
        num_heads=4,
    )


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 32, 32)`` for VibeToken."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "VibeToken (fixed-length 1D image tokenizer)",
        "build_vibetoken",
        "example_input",
        "2026",
        "DC",
    ),
]
