"""MWT: Multiwavelet-based Operator Learning.

Gupta et al., 2021.
Paper: https://arxiv.org/abs/2109.13459
Source: https://github.com/gaurav71531/mwt-operator

MWT replaces FNO's Fourier spectral layers with multiwavelet decomposition layers.
The key primitive: a 1D/2D multiwavelet transform layer that:
  1. Decomposes input into a low-pass (scaling) channel and high-pass (wavelet) channels
     using learnable filter matrices L (low-pass) and H (high-pass)
  2. Applies a learned linear operator in the wavelet coefficient space
     (four dense weight matrices for the LL, LH, HL, HH sub-bands for 2D)
  3. Reconstructs via the inverse multiwavelet transform

The full model stacks several MWT layers (each with decomposition+operator+reconstruction)
followed by a pointwise output decoder.

Simplification: 1D version with 2 MWT layers; scale (filter) factor = 2 modes;
input is a function on 64 points. The learnable filter bank uses orthogonal-init
matrices L, H of shape (k, k) where k is the number of multiwavelet channels.
This is the distinctive primitive; backbone: simple projections.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _orthogonal_init(shape: tuple) -> nn.Parameter:
    """Orthogonal init for filter matrices."""
    t = torch.empty(*shape)
    nn.init.orthogonal_(t)
    return nn.Parameter(t)


class MWTFilter(nn.Module):
    """Learnable multiwavelet analysis/synthesis filters.

    For a k-channel multiwavelet, L and H are (k,k) matrices that split
    an input vector of k coefficients into k low-pass and k high-pass outputs.
    """

    def __init__(self, k: int = 4) -> None:
        super().__init__()
        self.k = k
        # Analysis (forward) filters
        self.L = _orthogonal_init((k, k))  # low-pass
        self.H = _orthogonal_init((k, k))  # high-pass
        # Synthesis (inverse) filters
        self.Linv = _orthogonal_init((k, k))
        self.Hinv = _orthogonal_init((k, k))


class MWTLayer1D(nn.Module):
    """Single multiwavelet operator layer for 1D functions.

    Input: (B, C, N) function values where C = k (multiwavelet channels).
    Operation:
      1. Split into even/odd sub-sequences (Haar-like halving)
      2. Apply L, H matrices -> low-pass and high-pass at half resolution
      3. Apply learned operator in coefficient space (W_LL on low, W_HH on high)
      4. Reconstruct via Linv, Hinv
    """

    def __init__(self, k: int = 4, levels: int = 2) -> None:
        super().__init__()
        self.k = k
        self.levels = levels
        self.filters = MWTFilter(k)

        # Learned operators in coefficient space: one for each level, per subband
        # W_L: operator on low-pass, W_H: operator on high-pass at each level
        self.W_L = nn.ParameterList([nn.Parameter(torch.randn(k, k) * 0.1) for _ in range(levels)])
        self.W_H = nn.ParameterList([nn.Parameter(torch.randn(k, k) * 0.1) for _ in range(levels)])

    def _pad_to_even(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) % 2 != 0:
            x = F.pad(x, (0, 1))
        return x

    def _decompose(self, x: torch.Tensor):
        """One-level wavelet decomposition along last dim."""
        x = self._pad_to_even(x)
        N2 = x.size(-1) // 2
        xe = x[..., 0::2]  # even
        xo = x[..., 1::2]  # odd
        # Apply L and H (matrix multiply along channel dim)
        low = torch.einsum("ij,bjn->bin", self.filters.L, xe) + torch.einsum(
            "ij,bjn->bin", self.filters.L, xo
        )
        high = torch.einsum("ij,bjn->bin", self.filters.H, xe) - torch.einsum(
            "ij,bjn->bin", self.filters.H, xo
        )
        return low, high

    def _reconstruct(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """One-level wavelet reconstruction."""
        B, C, N2 = low.shape
        N = N2 * 2
        xe = torch.einsum("ij,bjn->bin", self.filters.Linv, low) + torch.einsum(
            "ij,bjn->bin", self.filters.Hinv, high
        )
        xo = torch.einsum("ij,bjn->bin", self.filters.Linv, low) - torch.einsum(
            "ij,bjn->bin", self.filters.Hinv, high
        )
        out = torch.zeros(B, C, N, device=low.device, dtype=low.dtype)
        out[..., 0::2] = xe
        out[..., 1::2] = xo
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        low_levels = []
        cur = x
        for lv in range(self.levels):
            low, high = self._decompose(cur)
            # Apply spectral operator on high-pass at this level
            high = torch.einsum("ij,bjn->bin", self.W_H[lv], high)
            low_levels.append((low, high))
            cur = low

        # Apply operator on coarsest low-pass
        coarse = torch.einsum("ij,bjn->bin", self.W_L[-1], cur)

        # Reconstruct from coarsest to finest
        rec = coarse
        for lv in reversed(range(self.levels)):
            low_in, high_in = low_levels[lv]
            # Pad rec to match high_in size if needed
            if rec.size(-1) < high_in.size(-1):
                rec = F.pad(rec, (0, high_in.size(-1) - rec.size(-1)))
            rec = self._reconstruct(rec, high_in)
            if rec.size(-1) > x.size(-1):
                rec = rec[..., : x.size(-1)]

        return rec


class MWTOperator1D(nn.Module):
    """Full MWT operator network for 1D PDE operator learning.

    Stacks MWT layers with pointwise residual connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        k: int = 4,  # multiwavelet channels
        n_layers: int = 2,
        levels: int = 2,
        width: int = 16,  # lifting width
    ) -> None:
        super().__init__()
        self.lift = nn.Conv1d(in_channels, k, kernel_size=1)
        self.mwt_layers = nn.ModuleList([MWTLayer1D(k, levels) for _ in range(n_layers)])
        self.w_bypass = nn.ModuleList([nn.Conv1d(k, k, kernel_size=1) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(k, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, N)
        h = self.lift(x)
        for mwt, w in zip(self.mwt_layers, self.w_bypass):
            h = h + mwt(h) + w(h)
        return self.project(h)


def build_mwt() -> nn.Module:
    return MWTOperator1D(in_channels=1, out_channels=1, k=4, n_layers=2, levels=2, width=16)


def example_input_mwt() -> torch.Tensor:
    # (B=1, C=1, N=64): 1D function on 64 grid points
    return torch.randn(1, 1, 64)


MENAGERIE_ENTRIES = [
    ("MWT (Multiwavelet Transform Operator)", "build_mwt", "example_input_mwt", "2021", "DC"),
]
