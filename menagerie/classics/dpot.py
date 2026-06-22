"""DPOT: Denoising Pre-training Operator Transformer.

Herde et al., 2024 (NeurIPS 2024).
Paper: https://arxiv.org/abs/2403.03542
Source: https://github.com/HelmholtzAI-FZJ/DPOT (alternative);
        official: https://github.com/shuhao02/DPOT

DPOT is a pre-trained PDE foundation model. Its key architectural primitive:

1. PATCH TOKENIZATION: the 2D spatial field is split into non-overlapping patches
   and each patch (with coordinates) is projected to a d_model embedding. This
   produces a sequence of patch tokens with 2D positional encodings.

2. FOURIER-ENHANCED ATTENTION: the transformer attention uses a FOURIER-ATTENTION
   variant where the attention weights are augmented with Fourier-mode mixing.
   Specifically, an additional "spectral" pathway applies rfft2 -> truncate modes ->
   linear transform in frequency -> irfft2 on the token sequence (interpreted as a
   2D grid), and this is added to the standard attention output. This gives
   multi-scale spatial mixing.

3. DENOISING PRE-TRAINING OBJECTIVE: during training, Gaussian noise is added to
   the input field tokens and the model is trained to denoise them (like DDPM/MAE).
   The architecture is otherwise identical to a standard transformer.

Here we implement the inference-time architecture: patch tokenizer + Fourier-attention
transformer blocks + patch decoder.

Simplifications: 4x4 patches on a 16x16 grid (16 patch tokens), 2 transformer blocks,
d_model=64, 4 heads; Fourier attention with 4 modes; no pre-training, random init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer2D(nn.Module):
    """Split 2D field into patches and project to d_model tokens.

    Input: (B, C_in, H, W) field
    Output: (B, n_patches, d_model) tokens + (n_patches,) 2D pos encodings
    """

    def __init__(self, patch_size: int, in_channels: int, d_model: int, H: int, W: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        n_patches = self.n_h * self.n_w
        patch_dim = in_channels * patch_size * patch_size + 2  # +2 for (x,y) center coords

        self.proj = nn.Linear(patch_dim, d_model)
        self.pos_emb = nn.Embedding(n_patches, d_model)
        self.register_buffer("pos_idx", torch.arange(n_patches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        # Extract patches: (B, C, n_h, p, n_w, p) -> (B, n_h, n_w, C*p*p)
        x = x.reshape(B, C, self.n_h, p, self.n_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_h * self.n_w, C * p * p)

        # Compute patch center coordinates (normalized)
        nh, nw = self.n_h, self.n_w
        row_coords = (torch.arange(nh, device=x.device).float() + 0.5) / nh
        col_coords = (torch.arange(nw, device=x.device).float() + 0.5) / nw
        # (n_h, n_w, 2)
        grid_y, grid_x = torch.meshgrid(row_coords, col_coords, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # (n_patches, 2)
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, n_patches, 2)

        # Concatenate patch features + coords
        patch_feat = torch.cat([x, coords], dim=-1)  # (B, n_patches, C*p*p + 2)
        tokens = self.proj(patch_feat)  # (B, n_patches, d_model)
        tokens = tokens + self.pos_emb(self.pos_idx)
        return tokens


class FourierAttentionLayer(nn.Module):
    """Fourier-enhanced attention: standard MHA + spectral mixing pathway.

    The spectral pathway interprets the n_patches token sequence as a 2D grid
    (n_h x n_w), applies rfft2 on the d_model features, truncates to n_modes,
    applies a learned complex linear, irfft2, then adds to the attention output.
    """

    def __init__(self, d_model: int, n_heads: int, n_h: int, n_w: int, n_modes: int = 4) -> None:
        super().__init__()
        self.n_h = n_h
        self.n_w = n_w
        self.n_modes = n_modes

        # Standard multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Spectral pathway: learned complex weights for Fourier modes
        # rfft2 of (n_h, n_w) field gives (n_h, n_w//2+1) complex; truncate
        mh = min(n_modes, n_h)
        mw = min(n_modes, n_w // 2 + 1)
        self.spec_re = nn.Parameter(torch.randn(mh, mw, d_model, d_model) * 0.02)
        self.spec_im = nn.Parameter(torch.randn(mh, mw, d_model, d_model) * 0.02)
        self.mh = mh
        self.mw = mw

    def _spectral_mix(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_h*n_w, d_model) -> reshape to 2D grid, rfft2, mix, irfft2
        B, N, D = x.shape
        h = x.reshape(B, self.n_h, self.n_w, D)

        # rfft2 along spatial dims (h, w)
        h_f = torch.fft.rfft2(h.permute(0, 3, 1, 2))  # (B, D, n_h, n_w//2+1)
        h_f = h_f.permute(0, 2, 3, 1)  # (B, n_h, n_w//2+1, D)

        # Truncate
        h_f_t = h_f[:, : self.mh, : self.mw, :]  # (B, mh, mw, D)

        # Complex linear: out = W_re * h_re - W_im * h_im + i*(W_re * h_im + W_im * h_re)
        h_re, h_im = h_f_t.real, h_f_t.imag
        out_re = torch.einsum("bhwd,hwde->bhwe", h_re, self.spec_re) - torch.einsum(
            "bhwd,hwde->bhwe", h_im, self.spec_im
        )
        out_im = torch.einsum("bhwd,hwde->bhwe", h_re, self.spec_im) + torch.einsum(
            "bhwd,hwde->bhwe", h_im, self.spec_re
        )

        # Pad back to full spectrum
        out_f = torch.zeros(
            B, self.n_h, self.n_w // 2 + 1, D, dtype=torch.complex64, device=x.device
        )
        out_f[:, : self.mh, : self.mw, :] = torch.complex(out_re, out_im)

        # irfft2
        out_f_p = out_f.permute(0, 3, 1, 2)  # (B, D, n_h, n_w//2+1)
        out_sp = torch.fft.irfft2(out_f_p, s=(self.n_h, self.n_w))  # (B, D, n_h, n_w)
        out_sp = out_sp.permute(0, 2, 3, 1).reshape(B, N, D)  # (B, N, D)
        return out_sp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard attention
        attn_out, _ = self.attn(x, x, x)
        # Spectral pathway
        spec_out = self._spectral_mix(x)
        return attn_out + spec_out


class DPOTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_h: int, n_w: int, n_modes: int = 4) -> None:
        super().__init__()
        self.fattn = FourierAttentionLayer(d_model, n_heads, n_h, n_w, n_modes)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.fattn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class PatchDecoder(nn.Module):
    """Project patch tokens back to spatial field."""

    def __init__(
        self, patch_size: int, out_channels: int, d_model: int, n_h: int, n_w: int
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_h = n_h
        self.n_w = n_w
        self.out_channels = out_channels
        self.proj = nn.Linear(d_model, out_channels * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_patches, d_model)
        B = tokens.size(0)
        p = self.patch_size
        C = self.out_channels
        patch_vals = self.proj(tokens)  # (B, n_patches, C*p*p)
        # Reshape to (B, C, H, W)
        patch_vals = patch_vals.reshape(B, self.n_h, self.n_w, C, p, p)
        # (B, C, n_h, p, n_w, p) -> (B, C, H, W)
        out = patch_vals.permute(0, 3, 1, 4, 2, 5).reshape(B, C, self.n_h * p, self.n_w * p)
        return out


class DPOT(nn.Module):
    """Denoising Pre-training Operator Transformer.

    Patch tokenizer -> Fourier-attention transformer -> patch decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        H: int = 16,
        W: int = 16,
        patch_size: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_modes: int = 4,
    ) -> None:
        super().__init__()
        n_h, n_w = H // patch_size, W // patch_size
        self.tokenizer = PatchTokenizer2D(patch_size, in_channels, d_model, H, W)
        self.blocks = nn.ModuleList(
            [DPOTBlock(d_model, n_heads, n_h, n_w, n_modes) for _ in range(n_layers)]
        )
        self.decoder = PatchDecoder(patch_size, out_channels, d_model, n_h, n_w)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        tokens = self.tokenizer(x)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.decoder(tokens)


def build_dpot() -> nn.Module:
    return DPOT(
        in_channels=1,
        out_channels=1,
        H=16,
        W=16,
        patch_size=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_modes=4,
    )


def example_input_dpot() -> torch.Tensor:
    # (B=1, C=1, H=16, W=16): 2D PDE field
    return torch.randn(1, 1, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "DPOT (Denoising Pretraining Operator Transformer)",
        "build_dpot",
        "example_input_dpot",
        "2024",
        "DC",
    ),
]
