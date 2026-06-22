"""Multiplicative Filter Networks (MFN) with residual connections.

Fathony et al., "Multiplicative Filter Networks."
arXiv:2012.10047 (ICLR 2021).
Source: https://github.com/boschresearch/multiplicative-filter-networks

Distinctive primitive:
  MFN replaces the standard neuron (linear + nonlinear activation) with a
  MULTIPLICATIVE FILTER recursion: each layer is a Hadamard (element-wise)
  product of a linear map of the PREVIOUS hidden state with a FILTER function
  of the INPUT coordinates:

    h_0 = F_0(x)                    [initial filter applied to input]
    h_k = (W_k h_{k-1} + b_k) * F_k(x)   for k = 1, ..., K

  where F_k(x) are SINUSOIDAL (FourierNet) or GABOR filters:
    FourierNet: F_k(x) = sin(W_f x + b_f)      (random Fourier features)
    GaborNet:   F_k(x) = exp(-0.5 * gamma^2 * ||x||^2) * sin(mu^T x)

  The "residual" variant adds a skip connection from h_0 to each h_k:
    h_k = (W_k h_{k-1} + b_k) * F_k(x) + alpha_k * h_0

  This is a class of IMPLICIT NEURAL REPRESENTATIONS (INRs) that learn
  continuous coordinate functions. Final output: linear(h_K).

Architecture here:
  - FourierNet variant (sinusoidal filters) + residual.
  - GaborNet variant.
  Both share the same recursion skeleton; filter initialization differs.

Faithful-compact simplifications:
  - 2D coordinate input (x, y) for simplicity (paper uses ND).
  - 4 filter layers.
  - Hidden width = 32 filters.
  - Output dim = 1 (scalar field, e.g. implicit SDF or NeRF density).
  - Batch of 16 2D query points.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFilter(nn.Module):
    """Random Fourier sinusoidal filter bank: F(x) = sin(W x + b).

    Filters are FIXED at init (random, not learned) in the original MFN.
    """

    def __init__(self, d_in: int, d_hidden: int, scale: float = 1.0) -> None:
        super().__init__()
        # Random frequency matrix and phase -- fixed (not trainable in paper)
        W = torch.randn(d_hidden, d_in) * scale
        b = torch.rand(d_hidden) * 2 * math.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) -> (B, d_hidden)"""
        return torch.sin(x @ self.W.t() + self.b)


class GaborFilter(nn.Module):
    """Gabor filter bank: F(x) = exp(-0.5 * gamma^2 * ||x||^2) * sin(mu^T x + phi).

    Frequency mu and bandwidth gamma are FIXED (random init, not learned).
    """

    def __init__(self, d_in: int, d_hidden: int, scale: float = 1.0) -> None:
        super().__init__()
        mu = torch.randn(d_hidden, d_in) * scale
        phi = torch.rand(d_hidden) * 2 * math.pi
        gamma = torch.ones(d_hidden)  # isotropic bandwidth
        self.register_buffer("mu", mu)
        self.register_buffer("phi", phi)
        self.register_buffer("gamma", gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) -> (B, d_hidden)"""
        # Gaussian envelope
        env = torch.exp(-0.5 * (self.gamma**2) * (x**2).sum(dim=-1, keepdim=True))
        sinusoid = torch.sin(x @ self.mu.t() + self.phi)
        return env * sinusoid  # (B, d_hidden)


class MFNLayer(nn.Module):
    """One MFN layer: (W h + b) * F(x)  (+  residual from h_0)."""

    def __init__(self, d_hidden: int, residual: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.residual = residual
        if residual:
            self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        h: torch.Tensor,  # (B, d_hidden)
        filter_out: torch.Tensor,  # (B, d_hidden) = F_k(x)
        h0: torch.Tensor | None = None,  # initial filter output for residual
    ) -> torch.Tensor:
        out = self.linear(h) * filter_out
        if self.residual and h0 is not None:
            out = out + self.alpha * h0
        return out


class FourierNet(nn.Module):
    """MFN-FourierNet: sinusoidal multiplicative filter network (with residuals)."""

    def __init__(
        self,
        d_in: int = 2,
        d_hidden: int = 32,
        d_out: int = 1,
        n_layers: int = 4,
        residual: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        # Separate filter for each layer (fixed random, not trained)
        self.filters = nn.ModuleList(
            [FourierFilter(d_in, d_hidden, scale) for _ in range(n_layers)]
        )
        self.layers = nn.ModuleList([MFNLayer(d_hidden, residual) for _ in range(n_layers - 1)])
        self.out = nn.Linear(d_hidden, d_out)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) -> (B, d_out)"""
        h0 = self.filters[0](x)  # initial filter
        h = h0
        for i, (layer, filt) in enumerate(zip(self.layers, self.filters[1:])):
            h = layer(h, filt(x), h0 if self.residual else None)
        return self.out(h)


class GaborNet(nn.Module):
    """MFN-GaborNet: Gabor-filter multiplicative filter network."""

    def __init__(
        self,
        d_in: int = 2,
        d_hidden: int = 32,
        d_out: int = 1,
        n_layers: int = 4,
        residual: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.filters = nn.ModuleList([GaborFilter(d_in, d_hidden, scale) for _ in range(n_layers)])
        self.layers = nn.ModuleList([MFNLayer(d_hidden, residual) for _ in range(n_layers - 1)])
        self.out = nn.Linear(d_hidden, d_out)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) -> (B, d_out)"""
        h0 = self.filters[0](x)
        h = h0
        for layer, filt in zip(self.layers, self.filters[1:]):
            h = layer(h, filt(x), h0 if self.residual else None)
        return self.out(h)


def build_fouriernet() -> nn.Module:
    return FourierNet(d_in=2, d_hidden=32, d_out=1, n_layers=4, residual=True)


def build_gabornet() -> nn.Module:
    return GaborNet(d_in=2, d_hidden=32, d_out=1, n_layers=4, residual=True)


def example_input_mfn() -> torch.Tensor:
    """16 2D coordinate queries (e.g. image pixels in [-1,1]^2)."""
    torch.manual_seed(12)
    return torch.rand(16, 2) * 2 - 1


MENAGERIE_ENTRIES = [
    (
        "FourierNet (Multiplicative Filter Network)",
        "build_fouriernet",
        "example_input_mfn",
        "2021",
        "DC",
    ),
    (
        "GaborNet (Multiplicative Filter Network)",
        "build_gabornet",
        "example_input_mfn",
        "2021",
        "DC",
    ),
]
