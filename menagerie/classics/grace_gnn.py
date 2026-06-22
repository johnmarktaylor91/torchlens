"""GRACE: Graph Atomic Cluster Expansion message-passing interatomic potential.

Bochkarev, Lysogorskiy, Drautz, "Graph Atomic Cluster Expansion for Semilocal
Interatomic Potentials", Phys. Rev. X 14, 021036 (2024) / arXiv:2311.16326.
Source: https://github.com/ICAMS/grace-tensorpotential (TF / tensorpotential).

GRACE "ceilings" in the source catalog because the production code is TensorFlow
/ tensorpotential (no standalone pip-installable PyTorch module) and the forward
consumes a batched neighbour-list graph, not a plain tensor.  This is a faithful
pure-torch reimplementation of the *core* GRACE/MACE-style message passing on a
small fixed atomic graph, consuming ``[Z (N,), edge_index (2,E), edge_vec (E,3)]``.

Faithful-core architecture (the essence of GRACE / the ACE B-basis nucleus):
  - **radial basis**: Bessel functions with the DimeNet polynomial envelope
    (exactly the layer-0 radial used by GRACE/MACE), projected to C channels.
  - **spherical harmonics** up to ``lmax=2`` on the edge unit vectors (9 real Y).
  - **atomic base** ``A[i,c,lm] = sum_{j in N(i)} R[e,c] * h[j] * Y[e,lm]`` -- the
    ACE 2-body density (permutation-invariant neighbour sum, equivariant under
    rotation like Y).
  - **cluster-expansion invariant** (the nu=2 / power-spectrum special case of
    the ACE B-basis = SOAP power spectrum): per-l contraction over m
    ``p[i,c,l] = sum_m A[i,c,l,m]^2`` -> rotation-invariant body-order-3 features.
  - **message passing**: A is rebuilt from neighbour node features each layer, so
    L layers reach L*r_cut (GRACE's semilocal range); residual MLP update.
  - **energy head**: per-species reference + per-layer readout -> sum -> total E.

Faithfully drops (documented): full Clebsch-Gordan coupling, equivariant L>0
messages, body orders > 3, the tensorpotential channel-coupling weights.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def _bessel_envelope_radial(r: torch.Tensor, n_radial: int, r_cut: float) -> torch.Tensor:
    """Bessel radial basis with the DimeNet p=5 polynomial envelope."""
    r = r.clamp_min(1e-6)
    k = torch.arange(1, n_radial + 1, device=r.device, dtype=r.dtype)  # (R,)
    bessel = (
        math.sqrt(2.0 / r_cut) * torch.sin(k[None, :] * math.pi * r[:, None] / r_cut) / r[:, None]
    )
    x = (r / r_cut).clamp(max=1.0)
    p = 5.0
    env = (
        1
        - ((p + 1) * (p + 2) / 2) * x**p
        + p * (p + 2) * x ** (p + 1)
        - (p * (p + 1) / 2) * x ** (p + 2)
    )
    return bessel * env[:, None]  # (E, R)


def _real_sph_harm_l2(rhat: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonics up to lmax=2 (9 functions) on unit vectors (E,3)."""
    x, y, z = rhat[:, 0], rhat[:, 1], rhat[:, 2]
    y00 = torch.full_like(x, 0.282095)
    y1m1 = 0.488603 * y
    y10 = 0.488603 * z
    y1p1 = 0.488603 * x
    y2m2 = 1.092548 * x * y
    y2m1 = 1.092548 * y * z
    y20 = 0.315392 * (3 * z * z - 1)
    y2p1 = 1.092548 * x * z
    y2p2 = 0.546274 * (x * x - y * y)
    return torch.stack([y00, y1m1, y10, y1p1, y2m2, y2m1, y20, y2p1, y2p2], dim=-1)  # (E,9)


def _power_spectrum(A: torch.Tensor) -> torch.Tensor:
    """Per-l m-contraction of the atomic base A (N,C,9) -> invariant (N,3C)."""
    p0 = (A[:, :, 0:1] ** 2).sum(-1)  # l=0
    p1 = (A[:, :, 1:4] ** 2).sum(-1)  # l=1
    p2 = (A[:, :, 4:9] ** 2).sum(-1)  # l=2
    return torch.cat([p0, p1, p2], dim=-1)  # (N, 3C)


class _GraceLayer(nn.Module):
    """One semilocal GRACE message-passing layer: build A from neighbours -> invariant -> update."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(channels + 3 * channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        R: torch.Tensor,
        Ysh: torch.Tensor,
    ):
        i, j = edge_index[0], edge_index[1]
        N, C = h.shape
        # edge feature: R[e,c] * h[j,c] outer Y[e,lm]  -> (E, C, 9)
        edge = (R * h[j]).unsqueeze(-1) * Ysh.unsqueeze(1)
        A = torch.zeros(N, C, Ysh.shape[1], device=h.device, dtype=h.dtype)
        A.index_add_(0, i, edge)  # neighbour sum -> atomic base (ACE 2-body density)
        inv = _power_spectrum(A)  # (N, 3C) invariant
        h = h + self.update(torch.cat([h, inv], dim=-1))  # residual update
        return h, A


class GraceGNN(nn.Module):
    """GRACE faithful-core: radial+SH edge embed -> ACE message passing -> per-atom energy."""

    def __init__(
        self,
        n_species: int = 8,
        channels: int = 32,
        n_layers: int = 2,
        n_radial: int = 8,
        r_cut: float = 5.0,
    ) -> None:
        super().__init__()
        self.r_cut = r_cut
        self.n_radial = n_radial
        self.species_embed = nn.Embedding(n_species, channels)
        self.radial_proj = nn.Linear(n_radial, channels, bias=False)
        self.layers = nn.ModuleList([_GraceLayer(channels) for _ in range(n_layers)])
        self.readouts = nn.ModuleList(
            [nn.Linear(channels, 1) for _ in range(n_layers - 1)]
            + [nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1))]
        )
        self.e_ref = nn.Embedding(n_species, 1)

    def forward(
        self, Z: torch.Tensor, edge_index: torch.Tensor, edge_vec: torch.Tensor
    ) -> torch.Tensor:
        r = edge_vec.norm(dim=-1)  # (E,)
        rhat = edge_vec / r.clamp_min(1e-6).unsqueeze(-1)
        Ysh = _real_sph_harm_l2(rhat)  # (E,9)
        R = self.radial_proj(_bessel_envelope_radial(r, self.n_radial, self.r_cut))  # (E,C)

        h = self.species_embed(Z)  # (N,C)
        e_atom = self.e_ref(Z).squeeze(-1)  # (N,) per-species reference
        for layer, readout in zip(self.layers, self.readouts):
            h, _ = layer(h, edge_index, R, Ysh)
            e_atom = e_atom + readout(h).squeeze(-1)  # hierarchical per-layer readout
        return e_atom.sum().unsqueeze(0)  # total energy (1,)


def build_grace() -> nn.Module:
    return GraceGNN(n_species=8, channels=32, n_layers=2, n_radial=8, r_cut=5.0)


def example_input_grace() -> List[torch.Tensor]:
    """``[Z (6,), edge_index (2,E), edge_vec (E,3)]`` -- a small 6-atom cluster."""
    n = 6
    pos = torch.randn(n, 3) * 1.5
    # build a neighbour list within r_cut
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (n,n,3)
    dist = diff.norm(dim=-1)
    mask = (dist < 5.0) & (dist > 1e-6)
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([dst, src], dim=0)  # (2,E) messages src->dst (i<-j)
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]]  # (E,3)
    Z = torch.randint(0, 8, (n,))
    return [Z, edge_index, edge_vec]


MENAGERIE_ENTRIES = [
    (
        "GRACE (graph atomic cluster expansion interatomic potential GNN)",
        "build_grace",
        "example_input_grace",
        "2024",
        "DC",
    ),
]
