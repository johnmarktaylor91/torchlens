"""VSR-Transformer: Lightweight Video Super-Resolution Transformer (x4).

Cao et al., arxiv 2021.
Paper: https://arxiv.org/abs/2106.06847
Source: https://github.com/caojiezhang/VSR-Transformer

Distinctive primitive: Spatio-temporal attention for video SR.
VSR-Transformer (also known as VSRT) processes a short video clip by:
  1. Patch embedding: project each (B, T, C, H, W) clip frame into a token
     sequence using a Conv2d patch embed.
  2. Spatio-temporal attention blocks: each block performs self-attention
     JOINTLY over spatial patches AND temporal frames — i.e., tokens from
     all T frames in the clip are concatenated into one sequence and attention
     is computed across both dimensions simultaneously.
  3. Reconstruction: the output tokens are reshaped to the reference frame
     and upsampled x4 via pixel-shuffle.

This is architecturally distinct from VRT (which uses window attention with
explicit 3-D windows and parallel warping).  VSRT uses global spatio-temporal
attention across the whole token sequence, making the temporal coupling
explicit in the attention matrix.

Compact: 2-frame clip, 4x4 patch size, embed_dim=32, 2 ST-attention blocks.
Input: (1, T=2, 3, 32, 32) clip.
Output: (1, 3, 128, 128) x4 SR of the reference (first) frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Patch embedding (per frame)
# -----------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Embed each frame into a sequence of patch tokens.

    (B*T, C, H, W) -> (B*T, num_patches, embed_dim)
    with num_patches = (H//patch_size) * (W//patch_size).
    """

    def __init__(self, in_ch: int = 3, embed_dim: int = 32, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W) -> (B*T, embed_dim, H/P, W/P)
        x = self.proj(x)
        B, C, H, W = x.shape
        # Flatten spatial: (B*T, H*W, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


# -----------------------------------------------------------------------
# Spatio-Temporal Self-Attention Block
# -----------------------------------------------------------------------


class STAttentionBlock(nn.Module):
    """Spatio-Temporal Attention block.

    Tokens from all T frames are concatenated along the sequence dimension
    before computing multi-head self-attention, giving each spatial patch
    direct access to corresponding patches across all frames.
    """

    def __init__(self, dim: int, num_heads: int = 2, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T*N, dim) spatio-temporal token sequence."""
        shortcut = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------
# VSRT model
# -----------------------------------------------------------------------


class VSRTransformerLightweight(nn.Module):
    """Lightweight VSR-Transformer with spatio-temporal attention (x4).

    Forward:
        x: (B, T, C_in, H, W) video clip.
    Returns:
        (B, 3, H*4, W*4) super-resolved reference frame (frame 0).
    """

    def __init__(
        self,
        in_ch: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        n_blocks: int = 2,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size)

        # Spatio-temporal attention blocks
        self.st_blocks = nn.ModuleList(
            [STAttentionBlock(embed_dim, num_heads) for _ in range(n_blocks)]
        )

        # Norm after transformer
        self.norm = nn.LayerNorm(embed_dim)

        # Reconstruction head: expand to pixel-shuffle input
        # patch_size^2 * 3 = upsampling target per patch
        # Here we upsample x4: produce 3 * (patch_size * 4)^2 / patch_size^2
        # = 3 * 16 channels per patch, then pixel-shuffle-4
        self.recon = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 3 * patch_size * patch_size * 16),
        )
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        P = self.patch_size
        pH, pW = H // P, W // P  # num patches per dim

        # Step 1: patch embed all frames
        x_bt = x.reshape(B * T, C, H, W)
        tokens = self.patch_embed(x_bt)  # (B*T, pH*pW, embed_dim)
        N = tokens.shape[1]  # pH * pW

        # Step 2: reshape to (B, T*N, embed_dim) for spatio-temporal attention
        tokens = tokens.reshape(B, T * N, self.embed_dim)

        # Step 3: ST-attention blocks
        for blk in self.st_blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Step 4: extract reference-frame tokens (first T=0 frame)
        ref_tokens = tokens[:, :N, :]  # (B, N, embed_dim)

        # Step 5: reconstruct HR frame via linear + pixel-shuffle
        # Output of recon per token: 3 * P^2 * 16 values
        # We treat each patch token as producing a (P*4) x (P*4) * 3 block.
        # Layout: reshape ref_tokens to (B, pH, pW, embed_dim), apply recon,
        # then fold into a pixel-shuffle-compatible layout.
        up = self.recon(ref_tokens)  # (B, N, 3*P^2*16)
        C_up = up.shape[-1]  # = 3 * P^2 * 16

        # -> (B, pH, pW, C_up) -> (B, C_up, pH, pW)
        up = up.reshape(B, pH, pW, C_up).permute(0, 3, 1, 2).contiguous()

        # Fold patch pixels P^2 back into spatial dims, then pixel_shuffle x4
        # (B, C_up, pH, pW) = (B, 3*16*P^2, pH, pW)
        # First expand P^2 -> spatial: (B, 3*16, pH*P, pW*P) via pixel_shuffle(P)
        up = F.pixel_shuffle(up, P)  # (B, 3*16, pH*P, pW*P) = (B,48,H,W)
        # Then pixel_shuffle x4 -> (B, 3, H*4, W*4)
        up = self.pixel_shuffle(up)
        return up  # (B, 3, H*4, W*4)


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_vsrt_lightweight_x4() -> nn.Module:
    return VSRTransformerLightweight(in_ch=3, embed_dim=32, patch_size=4, n_blocks=2, num_heads=2)


def example_input_vsrt() -> torch.Tensor:
    """2-frame clip (1, 2, 3, 32, 32)."""
    return torch.randn(1, 2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "VSR-Transformer (Cao 2021, spatio-temporal attention video SR x4)",
        "build_vsrt_lightweight_x4",
        "example_input_vsrt",
        "2021",
        "DC",
    ),
]
