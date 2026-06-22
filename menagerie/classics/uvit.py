"""U-ViT: All are Worth Words: A ViT Backbone for Diffusion Models.

Bao et al., CVPR 2023. arXiv:2209.12152
Source: https://github.com/baofff/U-ViT

U-ViT treats ALL inputs (image patches, time, class/text condition) as tokens,
and applies a ViT backbone with a critical modification: U-Net-like long skip
connections between SHALLOW and DEEP transformer blocks.

Architecture:
  1. Patch embedding of image: (B, C, H, W) -> (B, N_patches, d_model).
  2. Time token: sinusoidal timestep embedding -> (B, 1, d_model), prepended.
  3. Class/condition token: nn.Embedding -> (B, 1, d_model), prepended.
  4. First half of transformer blocks (encoder-side, "down").
  5. Long skip concatenation: for each encoder block output, store it;
     before the corresponding decoder block, concat [decoder_tokens, encoder_tokens]
     along the channel dim and project back to d_model.
  6. Second half of transformer blocks (decoder-side, "up").
  7. Final conv patch decoder: (B, N_patches, d_model) -> (B, C, H, W).

The LONG SKIP CONNECTION (step 5) is the signature U-ViT primitive:
  dec_in = cat([dec_tokens, enc_skip], dim=-1)  -- doubles channel dim
  dec_in = linear(dec_in)                        -- project back to d_model

Compact:
  - Input: (1, 4, 16, 16) latent, patch_size=2, d_model=128, depth=4.
  - uvit_small: d_model=64, depth=4.
  - uvit_diffusion: d_model=128, depth=4 (same as uvit_small/base, shows the U-skip structure).
  - All use 2 extra tokens (time + class): total N_tokens = N_patches + 2.
  - Long skip connects first depth//2 blocks to last depth//2 blocks.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal time embedding
# ============================================================


class TimestepEmbed(nn.Module):
    """Sinusoidal + MLP timestep embedding."""

    def __init__(self, d_model: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)  # (B, d_model)


# ============================================================
# Standard ViT block (no adaLN; U-ViT uses standard pre-norm)
# ============================================================


class ViTBlock(nn.Module):
    """Standard pre-norm ViT block (multi-head self-attention + MLP)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        d_ff = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(x_attn)
        x = x + self.ff(self.norm2(x))
        return x


# ============================================================
# U-ViT
# ============================================================


class UViT(nn.Module):
    """U-ViT: all-token ViT backbone with long skip connections for diffusion.

    Compact reimplementation, random init, forward-pass only.
    Input: tuple (latent, t, label) where:
      latent: (B, in_channels, H, W)
      t: (B,) timestep float
      label: (B,) class label int
    Output: (B, in_channels, H, W) noise/velocity prediction.
    """

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 2,
        in_channels: int = 4,
        d_model: int = 128,
        depth: int = 4,  # total transformer blocks; long skip connects first depth//2 to last depth//2
        n_heads: int = 4,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        assert depth % 2 == 0, "depth must be even for symmetric U-skip"
        self.depth = depth
        self.n_extra_tokens = 2  # time + class
        self.patch_size = patch_size
        self.in_channels = in_channels
        n_patches = (input_size // patch_size) ** 2
        self.n_patches_side = input_size // patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding (patches + 2 extra tokens)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches + self.n_extra_tokens, d_model) * 0.02
        )

        # Time + class tokens
        self.t_embed = TimestepEmbed(d_model)
        self.cls_embed = nn.Embedding(n_classes, d_model)

        # Encoder blocks (first half)
        n_enc = depth // 2
        self.enc_blocks = nn.ModuleList([ViTBlock(d_model, n_heads) for _ in range(n_enc)])

        # Middle block
        self.mid_block = ViTBlock(d_model, n_heads)

        # Long-skip projections: cat([dec, enc]) -> d_model
        self.skip_projs = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(n_enc)])

        # Decoder blocks (second half)
        self.dec_blocks = nn.ModuleList([ViTBlock(d_model, n_heads) for _ in range(n_enc)])

        # Final norm + patch decoder
        self.norm = nn.LayerNorm(d_model)
        self.patch_decode = nn.Linear(d_model, patch_size * patch_size * in_channels)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, patch_size^2 * C) -> (B, C, H, W)
        B, N, _ = x.shape
        p = self.patch_size
        c = self.in_channels
        h = w = self.n_patches_side
        x = x.reshape(B, h, w, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(B, c, h * p, w * p)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        latent, t, label = x
        B = latent.shape[0]

        # Patch tokens
        patches = self.patch_embed(latent).flatten(2).transpose(1, 2)  # (B, N, D)

        # Time + class extra tokens
        t_tok = self.t_embed(t).unsqueeze(1)  # (B, 1, D)
        c_tok = self.cls_embed(label).unsqueeze(1)  # (B, 1, D)

        # Concatenate all tokens
        tokens = torch.cat([t_tok, c_tok, patches], dim=1)  # (B, 2+N, D)
        tokens = tokens + self.pos_embed

        # Encoder (first half): save skip states
        skips = []
        for blk in self.enc_blocks:
            tokens = blk(tokens)
            skips.append(tokens)

        # Middle block
        tokens = self.mid_block(tokens)

        # Decoder (second half): long skip connections
        for i, (skip_proj, blk) in enumerate(zip(self.skip_projs, self.dec_blocks)):
            enc_skip = skips[-(i + 1)]  # corresponding encoder skip (reversed)
            # Long skip: concat decoder + encoder along channel dim, project back
            tokens = skip_proj(torch.cat([tokens, enc_skip], dim=-1))
            tokens = blk(tokens)

        # Norm + patch decode (use only patch tokens, strip extra tokens)
        tokens = self.norm(tokens)
        patch_tokens = tokens[:, self.n_extra_tokens :, :]  # (B, N, D)
        out = self.patch_decode(patch_tokens)  # (B, N, p^2*C)
        return self._unpatchify(out)  # (B, C, H, W)


# ============================================================
# Zero-arg builders and example inputs
# ============================================================


def build_uvit_small() -> nn.Module:
    return UViT(
        input_size=16, patch_size=2, in_channels=4, d_model=64, depth=4, n_heads=4, n_classes=10
    )


def build_uvit_diffusion() -> nn.Module:
    """Slightly larger U-ViT showing the full U-skip structure."""
    return UViT(
        input_size=16, patch_size=2, in_channels=4, d_model=128, depth=4, n_heads=4, n_classes=10
    )


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latent (1,4,16,16) + timestep (1,) + class label (1,)."""
    return (
        torch.randn(1, 4, 16, 16),
        torch.randint(0, 1000, (1,)).float(),
        torch.randint(0, 10, (1,)),
    )


MENAGERIE_ENTRIES = [
    (
        "U-ViT-Small (ViT backbone diffusion model with long skip connections between shallow/deep blocks)",
        "build_uvit_small",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "U-ViT Diffusion (all-token ViT + U-Net long skip connections for diffusion, d_model=128)",
        "build_uvit_diffusion",
        "example_input",
        "2023",
        "DC",
    ),
]
