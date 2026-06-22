"""MDT: Masked Diffusion Transformer for image generation.

Gao et al. (2023), "Masked Diffusion Transformer is a Strong Image Synthesizer".
ICCV 2023.  arXiv:2303.14389.
Source: https://github.com/sail-sg/MDT

Distinctive primitives:
  1. DiT BLOCK (adaLN-Zero): standard Transformer block for latent diffusion, where
     the LayerNorm is replaced by ADAPTIVE LayerNorm with zero-initialised modulation.
     Conditioning vector c (from timestep + class label embeddings) is projected to
     (6 * d_model) to produce (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff,
     gate_ff).  Gate weights are initialised to zero -> block outputs zero at init,
     ensuring stable training.
  2. MASKING SCHEME: a fixed binary mask is applied to token positions; masked tokens
     receive a learnable mask token.  The "side-interpolater" is a small sub-decoder
     that reconstructs masked positions from unmasked ones.
  3. For the atlas: we use a FIXED mask (not random), and the side-interpolater is a
     single cross-attention layer (masked tokens query unmasked features).

Compact config: patch_size=2, img_size=8, d_model=32, n_heads=2, n_layers=2, n_classes=10.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Sinusoidal timestep embedding
# ==============================================================


def timestep_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ==============================================================
# DiT adaLN-Zero block
# ==============================================================


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning.

    c -> Linear -> (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m)
    attn and mlp outputs are scaled by gate (zero-initialised gates).
    """

    def __init__(self, d_model: int = 32, n_heads: int = 2, d_ff: int = 64) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        # adaLN modulation: 6 scalars per token
        self.adaLN_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the linear layer -> gates start at 0
        nn.init.zeros_(self.adaLN_proj[1].weight)
        nn.init.zeros_(self.adaLN_proj[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model), c: (B, d_model) -> (B, N, d_model)"""
        mod = self.adaLN_proj(c).unsqueeze(1).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod

        # Attention with adaLN
        h = self.norm1(x) * (1 + scale_a) + shift_a
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_a * attn_out

        # FFN with adaLN
        h = self.norm2(x) * (1 + scale_m) + shift_m
        h = self.ff2(F.gelu(self.ff1(h)))
        x = x + gate_m * h
        return x


# ==============================================================
# Side-interpolater: cross-attention (masked tokens -> unmasked features)
# ==============================================================


class SideInterpolater(nn.Module):
    """Reconstruct masked token positions by cross-attending to unmasked tokens."""

    def __init__(self, d_model: int = 32, n_heads: int = 2) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, d), mask: (N,) bool where True=masked.
        Replace masked tokens with mask_token + cross-attend to unmasked.
        Returns (B, N, d) with masked positions refined.
        """
        B, N, d = tokens.shape
        # Replace masked positions
        tokens_with_mask = tokens.clone()
        tokens_with_mask[:, mask, :] = self.mask_token.expand(B, mask.sum().item(), d)
        # Masked tokens as queries, unmasked as keys/values
        unmasked = tokens[:, ~mask, :]  # (B, n_unmasked, d)
        masked_q = tokens_with_mask[:, mask, :]  # (B, n_masked, d)
        refined, _ = self.cross_attn(
            self.norm_q(masked_q),
            self.norm_kv(unmasked),
            self.norm_kv(unmasked),
        )
        tokens_with_mask[:, mask, :] = tokens_with_mask[:, mask, :] + refined
        return tokens_with_mask


# ==============================================================
# MDT model
# ==============================================================


class MDTSmall(nn.Module):
    """Compact MDT: patchify + mask + DiT blocks + side-interpolater + unpatchify.

    Input: (B, 4) packed: [img_flat(1), timestep(1), class_label(1), unused(1)]
    Actually we take (B, patch_tokens * d_model + 2) = latent patches + [t, y].
    The model uses a FIXED mask for half the tokens.

    Output: (B, n_patches * d_model) denoised token sequence.
    """

    def __init__(
        self,
        n_patches: int = 16,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
        n_classes: int = 10,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.d_model = d_model
        self.n_classes = n_classes
        # Fixed mask: first half masked
        n_masked = max(1, int(n_patches * mask_ratio))
        mask = torch.zeros(n_patches, dtype=torch.bool)
        mask[:n_masked] = True
        self.register_buffer("mask", mask)

        # Patchify: latent tokens -> linear proj
        self.patch_proj = nn.Linear(d_model, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.class_emb = nn.Embedding(n_classes + 1, d_model)  # +1 for unconditional

        # DiT blocks (process only unmasked tokens)
        self.dit_blocks = nn.ModuleList([DiTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

        # Side-interpolater
        self.side_interp = SideInterpolater(d_model, n_heads)

        # Final norm + output
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches * d_model + 2) packed latent tokens + [timestep, class].
        Output: (B, n_patches * d_model) denoised tokens.
        """
        B = x.shape[0]
        d = self.d_model
        N = self.n_patches
        tokens_flat = x[:, : N * d]
        meta = x[:, N * d :]
        t = meta[:, 0].long().clamp(0, 999)
        y = meta[:, 1].long().clamp(0, self.n_classes - 1)

        tokens = tokens_flat.view(B, N, d)  # (B, N, d)
        tokens = self.patch_proj(tokens) + self.pos_emb

        # Conditioning
        t_emb = self.t_mlp(timestep_embedding(t, d))  # (B, d)
        c_emb = self.class_emb(y)  # (B, d)
        cond = t_emb + c_emb  # (B, d)

        mask = self.mask  # (N,) bool
        # Process all tokens through DiT (MDT can process all; masking acts during loss)
        h = tokens
        for blk in self.dit_blocks:
            h = blk(h, cond)

        # Side-interpolater: refine masked positions
        h = self.side_interp(h, mask)

        # Final adaLN
        mod = self.adaLN_final(cond)  # (B, 2d)
        shift, scale = mod.unsqueeze(1).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale) + shift
        h = self.out_proj(h)
        return h.reshape(B, -1)  # (B, N*d)


def build_mdt_small() -> nn.Module:
    return MDTSmall(n_patches=16, d_model=32, n_heads=2, n_layers=2).eval()


def example_input() -> torch.Tensor:
    """(1, 16*32 + 2) = (1, 514) packed: 16 latent tokens + [timestep=100, class=5]."""
    x = torch.randn(1, 16 * 32 + 2)
    x[0, -2] = 100.0  # timestep
    x[0, -1] = 5.0  # class
    return x


MENAGERIE_ENTRIES = [
    (
        "MDT (Masked Diffusion Transformer: adaLN-Zero DiT blocks + side-interpolater for masked latent tokens)",
        "build_mdt_small",
        "example_input",
        "2023",
        "DC",
    ),
]
