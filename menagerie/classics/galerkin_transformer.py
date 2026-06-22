"""Galerkin Transformer: Galerkin-type and Fourier-type linear attention for PDEs.

Cao 2021.
Paper: https://arxiv.org/abs/2105.14995
Source: https://github.com/scaomath/galerkin-transformer

The key primitive: Galerkin-type attention replaces softmax(QK^T)V with
Q(K^T V) / n  -- linear-complexity attention where K and V are layer-normed
(not Q). This is the Galerkin projection: project V onto the span of K using
Q as coordinates. No softmax; layer-norm on K and V (not on Q) stabilises the
attention without normalising the sequence dimension.

Fourier-type attention is the transpose: normalise Q and K, compute (Q^T K) V / n.

Two encoder variants reproduced here:
  - GalerkinTransformer: Galerkin-type linear attention + FFN
  - FourierTransformer: Fourier-type linear attention + FFN

Both use the same positional encoding (learnable per grid-node) and share the
same layer structure. Used for 2D PDE operator learning.

Simplifications: 2 layers, d_model=64, single head (the paper uses 4-8 heads
but the primitives are identical per head). Grid size 16x16 -> seq 256.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GalerkinAttention(nn.Module):
    """Galerkin-type linear attention: Q(K^T V)/n with LayerNorm on K,V."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.ln_k = nn.LayerNorm(self.d_head)
        self.ln_v = nn.LayerNorm(self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.Wq(x).view(B, N, H, Dh).transpose(1, 2)  # (B,H,N,Dh)
        K = self.Wk(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.Wv(x).view(B, N, H, Dh).transpose(1, 2)

        # Layer-norm on K and V (not Q) -- the Galerkin stabilisation
        K = self.ln_k(K)  # (B,H,N,Dh)
        V = self.ln_v(V)

        # Galerkin: Q (K^T V) / N   -- O(N) not O(N^2)
        KtV = torch.einsum("bhnd,bhnv->bhdv", K, V) / N  # (B,H,Dh,Dh)
        out = torch.einsum("bhnd,bhdv->bhnv", Q, KtV)  # (B,H,N,Dh)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class FourierAttention(nn.Module):
    """Fourier-type linear attention: (Q^T K) V / n with LayerNorm on Q,K."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.ln_q = nn.LayerNorm(self.d_head)
        self.ln_k = nn.LayerNorm(self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.Wq(x).view(B, N, H, Dh).transpose(1, 2)
        K = self.Wk(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.Wv(x).view(B, N, H, Dh).transpose(1, 2)

        # Layer-norm on Q and K (not V)
        Q = self.ln_q(Q)
        K = self.ln_k(K)

        # Fourier: (Q^T K) V / N
        QtK = torch.einsum("bhnd,bhne->bhde", Q, K) / N  # (B,H,Dh,Dh)
        out = torch.einsum("bhde,bhne->bhnd", QtK, V)  # (B,H,N,Dh)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GalerkinTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = GalerkinAttention(d_model, n_heads)
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FourierTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = FourierAttention(d_model, n_heads)
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GalerkinTransformer(nn.Module):
    """Galerkin-type linear attention operator transformer.

    Input: (B, N, in_channels) function values on a grid.
    Output: (B, N, out_channels) -- pointwise PDE solution.
    """

    def __init__(
        self,
        in_channels: int = 3,  # (u, x, y) -- function + coordinates
        out_channels: int = 1,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Embedding(grid_size * grid_size, d_model)
        self.blocks = nn.ModuleList(
            [GalerkinTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )
        self.grid_size = grid_size
        self._register_grid_positions()

    def _register_grid_positions(self) -> None:
        g = self.grid_size
        idx = torch.arange(g * g)
        self.register_buffer("grid_idx", idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_channels)
        h = self.input_proj(x) + self.pos_emb(self.grid_idx)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


class FourierTransformer(nn.Module):
    """Fourier-type linear attention operator transformer (companion to Galerkin)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Embedding(grid_size * grid_size, d_model)
        self.blocks = nn.ModuleList(
            [FourierTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )
        self.grid_size = grid_size
        self._register_grid_positions()

    def _register_grid_positions(self) -> None:
        g = self.grid_size
        idx = torch.arange(g * g)
        self.register_buffer("grid_idx", idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_emb(self.grid_idx)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


def build_galerkin_transformer() -> nn.Module:
    return GalerkinTransformer(
        in_channels=3, out_channels=1, d_model=64, n_layers=2, n_heads=4, grid_size=16
    )


def example_input_galerkin() -> torch.Tensor:
    # (B=1, N=256, in_channels=3): function value u + 2D coords (x,y) at 16x16 grid
    return torch.randn(1, 256, 3)


def build_fourier_transformer() -> nn.Module:
    return FourierTransformer(
        in_channels=3, out_channels=1, d_model=64, n_layers=2, n_heads=4, grid_size=16
    )


def example_input_fourier() -> torch.Tensor:
    return torch.randn(1, 256, 3)


MENAGERIE_ENTRIES = [
    ("Galerkin Transformer", "build_galerkin_transformer", "example_input_galerkin", "2021", "DC"),
    (
        "Fourier Transformer (Cao)",
        "build_fourier_transformer",
        "example_input_fourier",
        "2021",
        "DC",
    ),
]
