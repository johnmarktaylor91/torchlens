"""SiT: Scalable Interpolant Transformer.

Ma et al., 2024. arXiv:2401.08740
Source: https://github.com/willisma/SiT

SiT is architecturally nearly identical to DiT (Peebles & Xie 2023, arXiv:2212.09748),
replacing the DDPM diffusion objective with a stochastic-interpolant / flow-matching
objective at training time.  The network architecture itself is:

  1. Patch embedding: patchify (B, C, H, W) latent -> (B, N_patches, d_model).
  2. Timestep embedding: sinusoidal + MLP -> t_emb (B, d_model).
  3. Class embedding: nn.Embedding(n_classes, d_model) -> cls_emb (B, d_model).
  4. Conditioning: c = t_emb + cls_emb -> (B, d_model).
  5. adaLN-Zero blocks: each block uses c to predict 6 scale/shift/gate parameters
     (alpha_1, beta_1, alpha_2, beta_2, gamma_1, gamma_2) via a SiLU+Linear.
     The block applies: pre-attn scale/shift -> self-attn -> gate -> residual;
                        pre-FFN scale/shift -> FFN -> gate -> residual.
  6. Final normalization: adaLN-Zero final norm (scale/shift from c).
  7. Unpatchify: linear -> (B, N_patches, patch_H*patch_W*C) -> reshape to (B, C, H, W).

Distinctive primitives: timestep+class adaLN-Zero modulation + transformer blocks + unpatchify.

Compact:
  - Input: (1, 4, 16, 16) latent (4 channels, 16x16 spatial = 64 patches at patch_size=2).
  - sit:       d_model=128, n_heads=4, depth=2.
  - sit_small: d_model=64,  n_heads=4, depth=2.

NOTE: This implements the ARCHITECTURE (forward pass), not the training objective.
The stochastic-interpolant training distinguishes SiT from DiT; the network is the same.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal time embedding (same as DiT)
# ============================================================


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP -> d_model."""

    def __init__(self, d_model: int, freq_embed_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def _timestep_sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self._timestep_sinusoidal(t, self.freq_embed_size)
        return self.mlp(emb)


# ============================================================
# adaLN-Zero block (core SiT/DiT block)
# ============================================================


class SiTBlock(nn.Module):
    """adaLN-Zero transformer block.

    Given conditioning c (B, d_model), predicts 6 parameters:
      (shift_1, scale_1, gate_1, shift_2, scale_2, gate_2)
    applied as:
      x = x + gate_1 * attn(scale_1 * LN(x) + shift_1)
      x = x + gate_2 * ffn(scale_2 * LN(x) + shift_2)
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_proj = nn.Linear(d_model, d_model)

        d_ff = int(d_model * mlp_ratio)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # adaLN: SiLU + Linear -> 6 modulation params
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the adaLN output linear for stable training
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model), c: (B, d_model)
        params = self.adaLN(c).unsqueeze(1)  # (B, 1, 6*d_model)
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)

        # Self-attention with adaLN modulation
        normed = self.norm1(x) * (1 + scale1) + shift1
        B, N, D = normed.shape
        qkv = self.qkv(normed).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + gate1 * self.attn_proj(x_attn)

        # Feed-forward with adaLN modulation
        normed = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ff2(F.gelu(self.ff1(normed)))

        return x


# ============================================================
# Final adaLN layer norm + linear unpatchify
# ============================================================


class FinalLayer(nn.Module):
    """Final adaLN-Zero norm + linear to output channels."""

    def __init__(self, d_model: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.adaLN(c).unsqueeze(1)
        shift, scale = params.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


# ============================================================
# SiT main model
# ============================================================


class SiTModel(nn.Module):
    """SiT: Scalable Interpolant Transformer (random-init reimpl, forward-pass only).

    Architecturally equivalent to DiT; trained with stochastic-interpolant objective.
    Input: (B, C, H, W) latent + (B,) timestep + (B,) class label.
    Output: (B, C, H, W) velocity / noise prediction.
    """

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 2,
        in_channels: int = 4,
        d_model: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        n_patches = (input_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Timestep + class conditioning
        self.t_embed = TimestepEmbedder(d_model)
        self.cls_embed = nn.Embedding(n_classes, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([SiTBlock(d_model, n_heads) for _ in range(depth)])

        # Final layer
        self.final = FinalLayer(d_model, patch_size, in_channels)

        self.input_size = input_size
        self.n_patches_side = input_size // patch_size

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, patch_size^2 * C) -> (B, C, H, W)
        B, N, _ = x.shape
        p = self.patch_size
        c = self.in_channels
        h = w = self.n_patches_side
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(B, c, h * p, w * p)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        latent, t, label = x
        # Patchify
        patches = self.patch_embed(latent)  # (B, d_model, h/p, w/p)
        B, D, Hg, Wg = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self.pos_embed

        # Conditioning
        t_emb = self.t_embed(t)  # (B, d_model)
        c_emb = self.cls_embed(label)  # (B, d_model)
        c = t_emb + c_emb

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens, c)

        # Final layer + unpatchify
        tokens = self.final(tokens, c)
        return self._unpatchify(tokens)


# ============================================================
# Zero-arg builders
# ============================================================


def build_sit() -> nn.Module:
    return SiTModel(
        input_size=16, patch_size=2, in_channels=4, d_model=128, depth=2, n_heads=4, n_classes=10
    )


def build_sit_small() -> nn.Module:
    return SiTModel(
        input_size=16, patch_size=2, in_channels=4, d_model=64, depth=2, n_heads=4, n_classes=10
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
        "SiT (Scalable Interpolant Transformer, adaLN-Zero DiT backbone + interpolant objective)",
        "build_sit",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "SiT-Small (SiT with smaller d_model=64, adaLN-Zero + interpolant flow matching)",
        "build_sit_small",
        "example_input",
        "2024",
        "DC",
    ),
]
