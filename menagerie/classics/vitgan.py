"""ViTGAN: Vision Transformer GAN generator using implicit neural synthesis.

Lee et al., "ViTGAN: Training GANs with Vision Transformers",
ICLR 2022. arXiv:2107.04589.
Source: https://github.com/wilile26811249/ViTGAN

ViTGAN key contributions:
  1. TRANSFORMER GENERATOR: uses ViT-style transformer blocks to process latent tokens
     instead of convolutional upsampling. Latent z is expanded into a sequence of tokens,
     processed by transformer encoder blocks, then decoded to pixel RGB values.
  2. IMPLICIT NEURAL SYNTHESIS: each token corresponds to a spatial position; a small
     MLP (one per output pixel or patch) decodes the token to RGB.
  3. CIPS-STYLE COORDINATE CONDITIONING: CIPS (Chan et al.) variant uses sinusoidal
     positional encoding of pixel coordinates combined with style code, passed through
     a SIREN-like MLP (sine activations) to predict per-pixel color. This is a fully
     coordinate-conditioned implicit neural representation.
  4. L2 attention regularization in discriminator (not shown in generator).

All models are compact random-init CPU models with tiny spatial sizes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared primitives
# ============================================================


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block: multi-head self-attention + FFN."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# MODULE 22: vitgan_generator
# ============================================================


class ViTGANGenerator(nn.Module):
    """ViTGAN Generator: z -> token sequence -> transformer -> per-patch MLP -> image.

    Architecture:
    1. Expand z to N tokens (project z, replicate + add learned pos embedding)
    2. Transformer encoder blocks process the token sequence
    3. Reshape tokens back to spatial grid
    4. Project each token to pixel/patch RGB via linear head
    Signature: transformer encoder over latent tokens + implicit decode to pixels.
    """

    def __init__(
        self, z_dim: int = 32, dim: int = 32, n_tokens: int = 16, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens  # number of patch tokens (e.g. 4x4 = 16)
        self.dim = dim
        # Project z to initial token features (replicated across all positions)
        self.z_proj = nn.Linear(z_dim, dim)
        # Learnable positional embedding for each token
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        # Transformer encoder
        self.transformer = nn.Sequential(
            TransformerBlock(dim, num_heads),
            TransformerBlock(dim, num_heads),
        )
        # Decode each token to patch RGB (3 pixels per token in compact form)
        self.to_rgb = nn.Linear(dim, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        # Expand z to N tokens: project then replicate
        z_feat = self.z_proj(z).unsqueeze(1).expand(-1, self.n_tokens, -1)  # (B, N, dim)
        tokens = z_feat + self.pos_embed.expand(B, -1, -1)
        # Transformer processes token sequence
        tokens = self.transformer(tokens)
        # Decode to RGB per token: reshape to (B, 3, sqrt(N), sqrt(N))
        s = int(self.n_tokens**0.5)  # spatial grid side
        rgb = self.to_rgb(tokens)  # (B, N, 3)
        rgb = rgb.view(B, s, s, 3).permute(0, 3, 1, 2)  # (B, 3, s, s)
        return torch.tanh(rgb)


def build_vitgan_generator() -> nn.Module:
    return ViTGANGenerator(z_dim=32, dim=32, n_tokens=16, num_heads=4)


def example_vitgan_generator() -> torch.Tensor:
    return torch.randn(1, 32)


# ============================================================
# MODULE 23: vitgan_vit_generator
# ============================================================


class ViTGANViTGenerator(nn.Module):
    """ViTGAN variant with explicit ViT encoder backbone feeding a decoder.

    Architecture:
    1. Expand z to N tokens + pos embed
    2. ViT-style TransformerEncoder processes tokens (as backbone)
    3. A decoder MLP per token maps from feature dim to patch pixels
    Shows the explicit backbone + decoder structure.
    """

    def __init__(
        self, z_dim: int = 32, dim: int = 32, n_tokens: int = 16, patch_size: int = 4
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim
        self.patch_size = patch_size
        # Project z to token features
        self.z_proj = nn.Linear(z_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        # ViT encoder backbone
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
            ),
            num_layers=2,
        )
        # Decoder MLP: maps each token to patch_size^2 * 3 pixels
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, patch_size * patch_size * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        s = int(self.n_tokens**0.5)
        # ViT encoder backbone
        z_feat = self.z_proj(z).unsqueeze(1).expand(-1, self.n_tokens, -1)
        tokens = z_feat + self.pos_embed.expand(B, -1, -1)
        tokens = self.encoder(tokens)  # (B, N, dim)
        # Decoder: each token -> patch pixels
        patches = self.decoder(tokens)  # (B, N, patch_size^2 * 3)
        # Fold patches into image
        p = self.patch_size
        patches = patches.view(B, s, s, 3, p, p)
        # (B, 3, s*p, s*p)
        img = patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 3, s * p, s * p)
        return torch.tanh(img)


def build_vitgan_vit_generator() -> nn.Module:
    return ViTGANViTGenerator(z_dim=32, dim=32, n_tokens=16, patch_size=2)


def example_vitgan_vit_generator() -> torch.Tensor:
    return torch.randn(1, 32)


# ============================================================
# MODULE 24: vitgan_cips_generator
# ============================================================


class SIRENLayer(nn.Module):
    """SIREN layer: linear + sine activation.

    Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions",
    NeurIPS 2020. arXiv:2006.09661.
    Signature: sin(w * (Wx + b)) activation for implicit coordinate-based synthesis.
    """

    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ViTGANCIPSGenerator(nn.Module):
    """CIPS-style coordinate-conditioned implicit generator.

    CoordGAN / CIPS: per-pixel prediction from (coordinate, style_code) via SIREN MLP.
    Architecture:
    1. Map z to style code w via small MLP
    2. Sample pixel grid coordinates (x, y in [-1, 1])
    3. For each pixel: concatenate fourier-encoded coords + style w
    4. Pass through SIREN MLP (sin activations) to predict RGB

    Signature: SIREN MLP over (pixel_coords + style_code) -> per-pixel color.
    Fully implicit: no spatial convolutions.
    """

    def __init__(
        self,
        z_dim: int = 32,
        style_dim: int = 16,
        hidden_dim: int = 32,
        out_H: int = 8,
        out_W: int = 8,
    ) -> None:
        super().__init__()
        self.out_H = out_H
        self.out_W = out_W
        # Mapping network: z -> style w
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
        )
        # Fourier positional encoding for 2D coords (sin + cos of frequencies)
        n_freqs = 4
        coord_dim = 2 * n_freqs * 2  # (x, y) each encoded as sin+cos of n_freqs
        self.n_freqs = n_freqs
        # SIREN MLP: coord_dim + style_dim -> hidden -> ... -> 3
        in_dim = coord_dim + style_dim
        self.siren1 = SIRENLayer(in_dim, hidden_dim)
        self.siren2 = SIRENLayer(hidden_dim, hidden_dim)
        self.to_rgb = nn.Linear(hidden_dim, 3)

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Fourier positional encoding for 2D coordinates."""
        freqs = 2.0 ** torch.arange(self.n_freqs, dtype=coords.dtype, device=coords.device)
        # coords: (..., 2), freqs: (n_freqs,)
        x = coords.unsqueeze(-1) * freqs  # (..., 2, n_freqs)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).view(*coords.shape[:-1], -1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        # Map z to style code
        w = self.mapping(z)  # (B, style_dim)
        # Create pixel grid coordinates in [-1, 1]
        ys = torch.linspace(-1, 1, self.out_H, device=z.device)
        xs = torch.linspace(-1, 1, self.out_W, device=z.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        # Fourier encode coordinates
        enc_coords = self._fourier_encode(coords)  # (B, H, W, coord_dim)
        # Expand style to spatial grid
        w_expand = w.unsqueeze(1).unsqueeze(1).expand(-1, self.out_H, self.out_W, -1)
        # Concatenate coords + style
        feat = torch.cat([enc_coords, w_expand], dim=-1)  # (B, H, W, coord_dim+style_dim)
        # SIREN MLP: per-pixel synthesis
        feat_flat = feat.view(B * self.out_H * self.out_W, -1)
        h = self.siren1(feat_flat)
        h = self.siren2(h)
        rgb = self.to_rgb(h).view(B, self.out_H, self.out_W, 3).permute(0, 3, 1, 2)
        return torch.tanh(rgb)


def build_vitgan_cips_generator() -> nn.Module:
    return ViTGANCIPSGenerator(z_dim=32, style_dim=16, hidden_dim=32, out_H=8, out_W=8)


def example_vitgan_cips_generator() -> torch.Tensor:
    return torch.randn(1, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "vitgan_generator",
        "build_vitgan_generator",
        "example_vitgan_generator",
        "2022",
        "DC",
    ),
    (
        "vitgan_vit_generator",
        "build_vitgan_vit_generator",
        "example_vitgan_vit_generator",
        "2022",
        "DC",
    ),
    (
        "vitgan_cips_generator",
        "build_vitgan_cips_generator",
        "example_vitgan_cips_generator",
        "2022",
        "DC",
    ),
]
