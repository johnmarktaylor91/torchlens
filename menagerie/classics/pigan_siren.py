"""pi-GAN: periodic implicit generative adversarial network with FiLM-SIREN.

Chan et al. (2021), "pi-GAN: Periodic Implicit Generative Adversarial Networks
for 3D-Aware Image Synthesis".  CVPR 2021.  arXiv:2012.00926.
Source: https://github.com/marcoamonteiro/pi-GAN

Distinctive primitives:
  1. SIREN (Sinusoidal Representation Network, Sitzmann et al. 2020 arXiv:2006.09661):
     implicit MLP where activations are sin(omega_0 * (Wx + b)).  The periodic
     nonlinearity enables representing fine detail in implicit fields.
  2. FiLM conditioning (SPADE-style): a MAPPING NETWORK (small MLP z -> gammas, betas)
     produces per-layer FREQUENCY SHIFTS (gamma) and PHASE SHIFTS (beta) for each SIREN
     layer: out = sin(gamma * (Wx + b) + beta).  This lets the latent code z steer the
     entire implicit field.
  3. 3D queries: the SIREN field takes (x, y, z, view_direction) as input and produces
     (color, density) -- a neural radiance field with FiLM-SIREN.

Two entries:
  pigan_embedding_siren128 = the SIREN backbone (d_hidden=128; here scaled to 32)
  pigan_implicit_generator3d = SIREN + mapping network queried over 3D points

Compact config: d_latent=32, d_hidden=32, n_siren_layers=4, n_points=8 (ray points).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Mapping Network: z -> per-layer FiLM parameters (gamma, beta)
# ==============================================================


class MappingNetwork(nn.Module):
    """Maps z (B, d_latent) to FiLM params for n_layers SIREN layers.

    Output: list of (gamma, beta) pairs, each (B, d_hidden).
    """

    def __init__(self, d_latent: int = 32, d_hidden: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(0.2),
        )
        # Per-layer gamma and beta projections
        self.gammas = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)])
        self.betas = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)])

    def forward(self, z: torch.Tensor):
        """z: (B, d_latent) -> list of (gamma, beta), each (B, d_hidden)"""
        h = self.net(z)
        film_params = [(g(h), b(h)) for g, b in zip(self.gammas, self.betas)]
        return film_params


# ==============================================================
# SIREN with FiLM conditioning
# ==============================================================


class FiLMSIRENLayer(nn.Module):
    """One FiLM-SIREN layer: sin(gamma * (Wx + b) + beta)."""

    def __init__(
        self, d_in: int, d_out: int, omega_0: float = 30.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.omega_0 = omega_0
        # SIREN weight init
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / d_in, 1.0 / d_in)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6.0 / d_in) / omega_0,
                    math.sqrt(6.0 / d_in) / omega_0,
                )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in), gamma/beta: (B, d_out) -> (..., d_out)"""
        # x may be (B, N, d_in) or (B, d_in)
        pre = self.linear(x)  # (..., d_out)
        # Broadcast FiLM params
        if pre.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, d_out)
            beta = beta.unsqueeze(1)
        return torch.sin(gamma * self.omega_0 * pre + beta)


class FiLMSIREN(nn.Module):
    """FiLM-conditioned SIREN implicit MLP.

    Takes 3D query points (B, N, 3+3=6: xyz + view_dir) conditioned by FiLM
    params and outputs (B, N, 4) = (R, G, B, density).
    """

    def __init__(
        self,
        d_in: int = 6,
        d_hidden: int = 32,
        n_layers: int = 4,
        d_out: int = 4,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            din = d_in if i == 0 else d_hidden
            self.layers.append(FiLMSIRENLayer(din, d_hidden, omega_0, is_first=(i == 0)))
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor, film_params: list) -> torch.Tensor:
        """x: (B, N, d_in), film_params: list[(gamma, beta)] -> (B, N, d_out)"""
        for i, layer in enumerate(self.layers):
            gamma, beta = film_params[i]
            x = layer(x, gamma, beta)
        return self.out(x)


# ==============================================================
# pi-GAN: embedding SIREN (backbone only)
# ==============================================================


class PiGANEmbeddingSIREN128(nn.Module):
    """pi-GAN SIREN backbone (embedding only): FiLM-SIREN + mapping network.

    Input: (B, N, 6) query points + (B, 32) latent code packed as one tensor.
    Wrapped: input is (B, N*6 + 32) concatenated float, unpacked inside.
    """

    def __init__(self, d_latent: int = 32, d_hidden: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.n_points = 8
        self.d_in = 6
        self.mapping = MappingNetwork(d_latent, d_hidden, n_layers)
        self.siren = FiLMSIREN(self.d_in, d_hidden, n_layers, d_out=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_points*d_in + d_latent) -> (B, n_points, 4)"""
        B = x.shape[0]
        pts = x[:, : self.n_points * self.d_in].view(B, self.n_points, self.d_in)
        z = x[:, self.n_points * self.d_in :]  # (B, d_latent)
        film = self.mapping(z)
        return self.siren(pts, film)


# ==============================================================
# pi-GAN: full implicit generator (mapping + FiLM-SIREN)
# ==============================================================


class PiGANImplicitGenerator3D(nn.Module):
    """pi-GAN 3D-aware generator: mapping network + FiLM-SIREN NeRF field.

    Same as above but explicitly named + can have volume-rendering placeholder.
    Input: same packed (B, n_points*d_in + d_latent) tensor.
    Output: (B, n_points, 4) = (RGB, density) per query point.
    """

    def __init__(self, d_latent: int = 32, d_hidden: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_points = 8
        self.d_in = 6
        self.mapping = MappingNetwork(d_latent, d_hidden, n_layers)
        self.siren = FiLMSIREN(self.d_in, d_hidden, n_layers, d_out=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_points*d_in + d_latent) -> (B, n_points, 4)"""
        B = x.shape[0]
        pts = x[:, : self.n_points * self.d_in].view(B, self.n_points, self.d_in)
        z = x[:, self.n_points * self.d_in :]
        film = self.mapping(z)
        out = self.siren(pts, film)  # (B, 8, 4)
        # Flatten to (B, 32) to return a single 2D tensor for tracing
        return out.reshape(B, -1)


def build_pigan_embedding_siren128() -> nn.Module:
    return PiGANEmbeddingSIREN128(d_latent=32, d_hidden=32, n_layers=4).eval()


def build_pigan_implicit_generator3d() -> nn.Module:
    return PiGANImplicitGenerator3D(d_latent=32, d_hidden=32, n_layers=4).eval()


def example_input() -> torch.Tensor:
    """(2, 80) -- 2 batch, 8 points * 6 dims + 32 latent."""
    return torch.randn(2, 8 * 6 + 32)


MENAGERIE_ENTRIES = [
    (
        "pi-GAN FiLM-SIREN128 (sinusoidal-activation implicit MLP backbone with FiLM conditioning)",
        "build_pigan_embedding_siren128",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "pi-GAN Implicit Generator 3D (mapping network + FiLM-SIREN for 3D-aware NeRF generation)",
        "build_pigan_implicit_generator3d",
        "example_input",
        "2021",
        "DC",
    ),
]
