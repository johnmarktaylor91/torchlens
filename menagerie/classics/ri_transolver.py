"""Transolver: A Fast Transformer Solver for PDEs on General Geometries.

Wu et al., ICML 2024.
Paper: https://arxiv.org/abs/2402.02366
Source: https://github.com/thuml/Transolver

Distinctive primitive -- PHYSICS-ATTENTION over learned slices:
  Standard attention over N mesh points is O(N^2). Transolver instead learns a
  soft assignment of each of the N mesh points to a small number M of
  "slices" (physical eigen-tokens / coherent sub-regions):

    w = softmax(Linear(x))                    # (B, N, M) slice weights
    slice_tokens = w^T @ x   (normalized)     # (B, M, D) M << N tokens
    attended = MHA(slice_tokens)              # cheap O(M^2) attention
    x_out = w @ attended                      # scatter back to N points

  So global physical interactions are modelled by attention among M slice
  tokens, then deslice-scattered back to the mesh. This makes a transformer
  PDE solver scale linearly in the number of mesh points.

Faithful compact random-init reimplementation: the slice-attention block,
the (coords + function-values) input projection, and the stacked-layer +
linear-head structure are reproduced; widths/depth/slice-count are small so
the unrolled trace draws quickly. arXiv:2402.02366.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsAttention(nn.Module):
    """Physics-Attention: mesh points -> learned slices -> attention -> scatter back."""

    def __init__(self, dim: int, n_slices: int = 16, heads: int = 4) -> None:
        super().__init__()
        self.slice_proj = nn.Linear(dim, n_slices)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        w = torch.softmax(self.slice_proj(x), dim=-1)  # (B, N, M)
        wt = w.transpose(1, 2)  # (B, M, N)
        # normalized aggregation of mesh points into M slice tokens
        denom = wt.sum(dim=-1, keepdim=True) + 1e-6
        slice_tokens = (wt @ x) / denom  # (B, M, D)
        attended, _ = self.attn(slice_tokens, slice_tokens, slice_tokens, need_weights=False)
        scattered = w @ attended  # (B, N, D) deslice
        return self.norm(x + scattered)


class TransolverLayer(nn.Module):
    def __init__(self, dim: int, n_slices: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.physics_attn = PhysicsAttention(dim, n_slices, heads)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.physics_attn(x)
        x = x + self.mlp(self.norm(x))
        return x


class Transolver(nn.Module):
    def __init__(
        self,
        space_dim: int = 3,
        fun_dim: int = 16,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 64,
        out_dim: int = 1,
        slice_num: int = 16,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(space_dim + fun_dim, d_model)
        self.layers = nn.ModuleList(
            [TransolverLayer(d_model, slice_num, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, fx: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # fx: (B, N, fun_dim) function values ; coords: (B, N, space_dim)
        h = self.proj(torch.cat([fx, coords], dim=-1))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def build_transolver() -> nn.Module:
    """Build a compact Transolver (Physics-Attention PDE solver)."""
    return Transolver(
        space_dim=3, fun_dim=16, n_layers=4, n_heads=4, d_model=64, out_dim=1, slice_num=16
    )


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Example (function-values, coords): ``((1, 256, 16), (1, 256, 3))``."""
    fx = torch.randn(1, 256, 16)
    coords = torch.randn(1, 256, 3)
    return fx, coords


MENAGERIE_ENTRIES = [
    (
        "Transolver (Physics-Attention over learned slices for PDE solving)",
        "build_transolver",
        "example_input",
        "2024",
        "DC",
    ),
]
