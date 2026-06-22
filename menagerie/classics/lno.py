"""LNO: Laplace Neural Operator.

Cao et al., 2023.
Paper: https://arxiv.org/abs/2303.10528
Source: https://github.com/qingkaikong/Laplace-Neural-Operator (reference);
        official: https://github.com/ShuhaoLii/LNO (ICML 2024 version)

LNO replaces FNO's Fourier transform with the LAPLACE TRANSFORM via a
POLE-RESIDUE parameterization of the spectral operator:

  L[f](s) = integral_0^inf f(t) e^{-st} dt

The continuous transfer function H(s) is parameterized as a sum of pole-residue pairs:
  H(s) = sum_k r_k / (s - p_k)

where poles p_k and residues r_k are LEARNED COMPLEX PARAMETERS.

This gives a stable rational-function spectral operator with explicit damping/oscillation
characteristics controlled by the pole locations.

The LNO layer:
  1. Compute L-transform of input at the learned pole frequencies: F_k = integral f(t) * phi_k(t) dt
     where phi_k are basis functions derived from poles (discretized Laplace basis)
  2. Apply residue weights to the pole-domain representation
  3. Map back to time/spatial domain via the inverse representation

In practice (discrete/finite-N version):
  - Poles p_k and residues r_k are learned complex parameters (n_poles total)
  - For input x at grid points t_n, compute: Z_k = sum_n x_n * exp(-p_k * t_n) * dt
    (discrete Laplace basis evaluation at pole k)
  - Output Y_n = sum_k r_k * Z_k * exp(p_k * t_n)
    (inverse: reconstruct from pole-residue weighted components)
  - This is the pole-residue integral transform layer

Simplifications: 2 LNO layers, n_poles=8, d_model=32 (multiple channels processed
independently), input sequence length 32. Real-valued computation via real/imag split.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoleResidueLaplace(nn.Module):
    """Pole-residue Laplace-domain spectral operator.

    For a 1D sequence x of length N:
      - Compute forward Laplace at K poles: Z_k = sum_n x_n * exp(-p_k * t_n) * dt
      - Apply residue weights: Z_k <- r_k * Z_k
      - Reconstruct: Y_n = Re[ sum_k Z_k * exp(p_k * t_n) ]

    Poles are constrained to have negative real part (stable Laplace basis):
      p_k = -exp(alpha_k) + i * omega_k
    """

    def __init__(self, n_poles: int = 8, n_channels: int = 32, seq_len: int = 32) -> None:
        super().__init__()
        self.n_poles = n_poles
        self.seq_len = seq_len
        self.n_channels = n_channels

        # Learnable pole parameters: log-decay (alpha) and frequency (omega)
        self.pole_alpha = nn.Parameter(torch.randn(n_poles) * 0.5)  # decay > 0
        self.pole_omega = nn.Parameter(torch.randn(n_poles) * 2.0)  # frequency

        # Learnable residues (complex: real + imaginary parts)
        # Shape: (n_channels, n_poles) -- per-channel residues
        self.res_re = nn.Parameter(torch.randn(n_channels, n_poles) * 0.1)
        self.res_im = nn.Parameter(torch.randn(n_channels, n_poles) * 0.1)

        # Time grid (fixed)
        t = torch.linspace(0, 1, seq_len)
        self.register_buffer("t_grid", t)

    def _compute_basis(self) -> torch.Tensor:
        """Compute the Laplace basis matrix: (seq_len, n_poles) complex."""
        # Poles: p_k = -exp(alpha_k) + i * omega_k  (stable: Re < 0)
        p_re = -torch.exp(self.pole_alpha)  # (n_poles,)
        p_im = self.pole_omega  # (n_poles,)

        # exp(-p_k * t_n) = exp(exp(alpha_k)*t_n) * [cos(-omega_k*t_n) + i sin(-omega_k*t_n)]
        # Forward Laplace basis: phi_nk = exp(-(p_re_k + i*p_im_k) * t_n)
        # = exp(-p_re_k * t_n) * [cos(p_im_k * t_n) - i*sin(p_im_k * t_n)]
        t = self.t_grid.unsqueeze(1)  # (N, 1)
        pr = p_re.unsqueeze(0)  # (1, K)
        pi = p_im.unsqueeze(0)  # (1, K)

        fwd_re = torch.exp(-pr * t) * torch.cos(-pi * t)  # (N, K)
        fwd_im = torch.exp(-pr * t) * torch.sin(-pi * t)  # (N, K)
        return fwd_re, fwd_im

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_channels, seq_len)
        B, C, N = x.shape
        dt = 1.0 / N

        fwd_re, fwd_im = self._compute_basis()  # (N, K)

        # Forward transform: Z_k = sum_n x_n * phi_nk * dt  (over n)
        # (B, C, N) x (N, K) -> (B, C, K)
        Z_re = torch.einsum("bcn,nk->bck", x, fwd_re) * dt
        Z_im = torch.einsum("bcn,nk->bck", x, fwd_im) * dt

        # Apply residue weights: (B, C, K) * (C, K)
        rr = self.res_re.unsqueeze(0)  # (1, C, K)
        ri = self.res_im.unsqueeze(0)
        # Complex multiplication: (Z_re + i*Z_im) * (res_re + i*res_im)
        ZW_re = Z_re * rr - Z_im * ri
        ZW_im = Z_re * ri + Z_im * rr

        # Inverse: Y_n = Re[ sum_k (ZW_re + i*ZW_im) * exp(p_k * t_n) ]
        # exp(p_k * t_n) = exp(p_re_k * t_n) * [cos(p_im_k*t_n) + i*sin(p_im_k*t_n)]
        p_re = -torch.exp(self.pole_alpha)
        p_im = self.pole_omega
        t = self.t_grid.unsqueeze(1)
        inv_re = torch.exp(p_re.unsqueeze(0) * t) * torch.cos(p_im.unsqueeze(0) * t)  # (N, K)
        inv_im = torch.exp(p_re.unsqueeze(0) * t) * torch.sin(p_im.unsqueeze(0) * t)

        # Re[ (ZW_re + i*ZW_im) * (inv_re + i*inv_im) ] = ZW_re*inv_re - ZW_im*inv_im
        # Sum over K poles: (B, C, K) x (N, K) -> (B, C, N)
        Y = torch.einsum("bck,nk->bcn", ZW_re, inv_re) - torch.einsum("bck,nk->bcn", ZW_im, inv_im)
        return Y


class LNOLayer(nn.Module):
    """LNO layer: pole-residue Laplace spectral operator + pointwise bypass."""

    def __init__(self, n_channels: int, seq_len: int, n_poles: int = 8) -> None:
        super().__init__()
        self.spectral = PoleResidueLaplace(n_poles, n_channels, seq_len)
        self.bypass = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(n_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class LNO(nn.Module):
    """Laplace Neural Operator for 1D time-series PDE operator learning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        n_layers: int = 2,
        n_poles: int = 8,
        seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.layers = nn.ModuleList([LNOLayer(width, seq_len, n_poles) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, N)
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h)
        return self.project(h)


def build_lno() -> nn.Module:
    return LNO(in_channels=1, out_channels=1, width=32, n_layers=2, n_poles=8, seq_len=32)


def example_input_lno() -> torch.Tensor:
    # (B=1, C=1, N=32): 1D time-series / spatial function
    return torch.randn(1, 1, 32)


MENAGERIE_ENTRIES = [
    ("LNO (Laplace Neural Operator)", "build_lno", "example_input_lno", "2023", "DC"),
]
