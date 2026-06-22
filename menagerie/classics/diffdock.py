"""DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking.

Corso, Stärk, Jing, Barzilay & Jaakkola, ICLR 2023.
Paper: https://arxiv.org/abs/2210.01776
Source: https://github.com/gcorso/DiffDock

DiffDock treats molecular docking as a diffusion process over ligand poses.
Given a protein-ligand complex, it learns a score model (and a confidence model)
that guides the reverse diffusion process over SE(3) x torsion-angle space.

The ARCHITECTURE-DEFINING component is the **SE(3)-equivariant tensor-field score
model**: a GNN where nodes carry irreducible representations (irreps) of SE(3)
(scalar + vector features) and edge messages are computed via equivariant tensor
products using spherical harmonics of the edge direction.

The network (TensorProductConvLayer in DiffDock source) is closely related to
e3nn-style equivariant networks (also see: SEGNN, TFN, SE(3)-Transformer):
  - Each node carries: s_i (scalar, l=0) and v_i (vector, l=1) features.
  - Edge features: scalar + spherical-harmonic encoding of edge direction (Y_l).
  - Message: equivariant tensor-product interaction between source irreps and
    edge SH, gated by a learned scalar network.
  - Aggregation: sum over neighbours.
  - Update: MLP(aggregated message + node feat) -> new irreps.

This reimplementation:
  - Avoids the e3nn library (not installed) and reproduces the architecture
    faithfully using DIRECT COMPUTATION of the l=0 and l=1 tensor-product
    components from the edge spherical harmonics.
  - Implements l=0 (scalar) and l=1 (3-vector) irreps only, matching the
    compact "lmax=1" setting used in the DiffDock binding-site GNN.
  - Reproduces: spherical harmonics Y0, Y1 from edge direction; tensor-product
    scalar-x-scalar, scalar-x-vector, vector-x-scalar interactions; gating by
    a scalar MLP (the 'radial network' in DiffDock); node update MLP.
  - Graph: 10 nodes (atoms), random edges.

Simplifications:
  - lmax = 1 only (l=0 + l=1 irreps).
  - No torsion-angle branch (the rotatable-bond part is outside the GNN).
  - Confidence model and score model share the same GNN architecture; we
    implement the score model (same forward pass, different output head).
  - No time-step embedding or protein-receptor graph (just the ligand subgraph
    to show the equivariant tensor-product primitive).
  - Compact: 2 GNN layers, hidden_dim=32, 10 nodes.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spherical harmonics: Y_0 and Y_1 components from unit edge directions
# ---------------------------------------------------------------------------


def _spherical_harmonics_l0(direction: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonic Y_0^0 (scalar, constant).

    Y_0^0 = 1/sqrt(4*pi) -- we use the unnormalised version: 1.
    direction: (E, 3) unit vectors (not used, just for API consistency).
    Returns: (E, 1).
    """
    return torch.ones(direction.shape[0], 1, device=direction.device, dtype=direction.dtype)


def _spherical_harmonics_l1(direction: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonics Y_1^{-1,0,1} from unit direction vectors.

    For l=1: Y_1^m = (y, z, x) component ordering (matching e3nn convention).
    direction: (E, 3) unit vectors.
    Returns: (E, 3).
    """
    # e3nn l=1 ordering: (y, z, x)
    return torch.stack([direction[:, 1], direction[:, 2], direction[:, 0]], dim=-1)


# ---------------------------------------------------------------------------
# Radial (distance) basis and MLP for edge scalar features
# ---------------------------------------------------------------------------


class RadialBasisEncoding(nn.Module):
    """Gaussian radial basis encoding of edge distances.

    Maps distance d -> (E, n_rbf) feature vector via Gaussian bumps.
    """

    def __init__(self, n_rbf: int = 16, d_min: float = 0.5, d_max: float = 5.0) -> None:
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 2.0 / (d_max - d_min) * n_rbf

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (E,) edge distances.  Returns (E, n_rbf)."""
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class RadialNetwork(nn.Module):
    """MLP mapping radial basis + edge scalar features -> per-tensor-product gate scalars."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SE(3)-equivariant tensor-product convolution layer (lmax=1)
# ---------------------------------------------------------------------------


class TensorProductConvLayer(nn.Module):
    """DiffDock-style equivariant tensor-product message-passing layer (lmax=1).

    Nodes carry: scalar features s_i (d_s,) and vector features v_i (d_v, 3).
    Edges carry: radial basis of distance (n_rbf,) and direction unit vector.

    Message computation (for source node j -> destination node i):
      Y0 = Y_0(edge_dir) -- (E, 1)
      Y1 = Y_1(edge_dir) -- (E, 3)

      Tensor-product of source irreps x edge SH (Clebsch-Gordan):
        l=0 x l=0 -> l=0:  s_j * Y0         ->  m_s0   (E, d_s)
        l=0 x l=1 -> l=1:  s_j * Y1         ->  m_v0   (E, d_s, 3) ~ scalar modulates vector
        l=1 x l=0 -> l=1:  v_j * Y0         ->  m_v1   (E, d_v, 3) ~ vector x scalar
        l=1 x l=1 -> l=0:  sum(v_j * Y1,-1) ->  m_s1   (E, d_v)   ~ dot product

      Gate each pathway by a scalar learned from radial network:
        g = RadialNetwork(edge_feat)  -- (E, n_gates)

      Aggregate (scatter sum over source nodes to destination):
        agg_s = sum_j(gate_0 * m_s0 + gate_1 * m_s1)  -- (N, d_s + d_v)
        agg_v = sum_j(gate_2 * m_v0 + gate_3 * m_v1)  -- (N, (d_s+d_v), 3)

      Update via MLP:
        s_new = MLP([s_i, agg_s])   -- (N, d_s)
        v_new = linear([agg_v])      -- (N, d_v, 3)  (equivariant: no activation)

    This faithfully captures the topology of DiffDock's TensorProductConvLayer.
    """

    def __init__(
        self,
        d_s: int,  # scalar feature dimension
        d_v: int,  # vector feature dimension (triples)
        n_rbf: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_v = d_v

        # Radial network: maps edge features -> gates for 4 tensor-product paths
        # We concatenate RBF and any scalar edge features; for simplicity use RBF only
        n_gates = d_s + d_v + d_s + d_v  # for each TP path: shape of output
        self.radial_net = RadialNetwork(n_rbf, n_gates, hidden_dim)

        # Node update MLPs
        d_agg_s = d_s + d_v  # aggregated scalar (from s*Y0 + v.Y1)
        d_agg_v = d_s + d_v  # aggregated vector channel count

        self.update_s_mlp = nn.Sequential(
            nn.Linear(d_s + d_agg_s, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_s),
        )
        # Vector update is equivariant (linear on channel dim, no activation)
        self.update_v_linear = nn.Linear(d_agg_v, d_v, bias=False)

        self.layer_norm_s = nn.LayerNorm(d_s)
        self.layer_norm_v = nn.LayerNorm(d_v)

    def forward(
        self,
        s: torch.Tensor,  # (N, d_s) scalar node features
        v: torch.Tensor,  # (N, d_v, 3) vector node features
        edge_index: torch.Tensor,  # (2, E) [src, dst]
        edge_rbf: torch.Tensor,  # (E, n_rbf) radial basis
        edge_dir: torch.Tensor,  # (E, 3) unit direction vectors (dst - src normalised)
    ) -> tuple:
        """Returns (s_new, v_new) with same shapes."""
        N = s.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # --- Spherical harmonics ---
        Y0 = _spherical_harmonics_l0(edge_dir)  # (E, 1)
        Y1 = _spherical_harmonics_l1(edge_dir)  # (E, 3)

        # --- Gate scalars from radial network ---
        gates = self.radial_net(edge_rbf)  # (E, d_s + d_v + d_s + d_v)
        g_s0 = gates[:, : self.d_s]  # (E, d_s)  for s_j * Y0
        g_v1 = gates[:, self.d_s : self.d_s + self.d_v]  # (E, d_v)  for v_j * Y0
        g_s1 = gates[:, self.d_s + self.d_v : 2 * self.d_s + self.d_v]  # (E, d_s) for s_j * Y1
        g_v0 = gates[:, 2 * self.d_s + self.d_v :]  # (E, d_v)  for v_j.Y1 -> scalar

        s_src = s[src]  # (E, d_s)
        v_src = v[src]  # (E, d_v, 3)

        # --- Tensor product messages ---
        # l=0 x l=0 -> l=0: m_s0 = gate * s_j * Y0
        m_s0 = g_s0 * s_src * Y0  # (E, d_s)

        # l=1 x l=1 -> l=0: m_s1 = gate * dot(v_j, Y1)
        # v_src: (E, d_v, 3), Y1: (E, 3)
        dot_vY1 = (v_src * Y1.unsqueeze(1)).sum(-1)  # (E, d_v)
        m_s1 = g_v0 * dot_vY1  # (E, d_v)

        # l=0 x l=1 -> l=1: m_v0 = gate * s_j * Y1 (broadcast s_j over 3 dims)
        m_v0 = g_s1.unsqueeze(-1) * s_src.unsqueeze(-1) * Y1.unsqueeze(1)  # (E, d_s, 3)

        # l=1 x l=0 -> l=1: m_v1 = gate * v_j * Y0
        m_v1 = g_v1.unsqueeze(-1) * v_src * Y0.unsqueeze(1)  # (E, d_v, 3)

        # --- Aggregate (scatter sum) ---
        agg_s = torch.zeros(N, self.d_s + self.d_v, device=s.device, dtype=s.dtype)
        agg_v = torch.zeros(N, self.d_s + self.d_v, 3, device=s.device, dtype=s.dtype)

        # Concatenate scalar aggregation channels
        m_s_all = torch.cat([m_s0, m_s1], dim=-1)  # (E, d_s+d_v)
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_s_all), m_s_all)

        # Concatenate vector aggregation channels
        m_v_all = torch.cat([m_v0, m_v1], dim=1)  # (E, d_s+d_v, 3)
        agg_v.scatter_add_(
            0,
            dst.unsqueeze(-1).unsqueeze(-1).expand_as(m_v_all),
            m_v_all,
        )

        # --- Node update ---
        s_new = self.layer_norm_s(s + self.update_s_mlp(torch.cat([s, agg_s], dim=-1)))
        # Equivariant vector update: linear on channel dim
        agg_v_ch = agg_v.view(N, (self.d_s + self.d_v) * 3)  # flatten channel+xyz
        agg_v_ch = agg_v_ch.view(N, self.d_s + self.d_v, 3)
        # Linear across channel dim (equivariant: same weights for each xyz component)
        v_new = v + self.update_v_linear(
            agg_v_ch.permute(0, 2, 1)  # (N, 3, d_s+d_v)
        ).permute(0, 2, 1)  # (N, d_v, 3)
        v_new = v_new / (v_new.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        v_new = v_new * self.layer_norm_v(v_new.norm(dim=-1)).unsqueeze(-1)

        return s_new, v_new


# ---------------------------------------------------------------------------
# DiffDock Score Model (equivariant GNN)
# ---------------------------------------------------------------------------


class DiffDockScoreModel(nn.Module):
    """DiffDock score model: SE(3)-equivariant GNN over the ligand graph.

    Maps node features (atom types) + edge geometry (distances, directions)
    through equivariant tensor-product convolution layers, then predicts:
      - tr_score: translation score (ligand CoM displacement, 3-vector)
      - rot_score: rotation score (axis-angle, 3-vector)

    The protein-receptor context (cross-graph attention) is omitted here;
    we show the equivariant ligand GNN primitive.
    """

    def __init__(
        self,
        n_atom_types: int = 9,
        d_s: int = 32,
        d_v: int = 8,
        n_rbf: int = 16,
        n_layers: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        # Atom embedding: type -> (d_s,)
        self.atom_emb = nn.Embedding(n_atom_types, d_s)
        # Initial vector features are zero (set from node embedding broadcast later)
        self.v_init = nn.Linear(d_s, d_v * 3)  # initialise vector features from scalars

        # Radial basis
        self.rbf = RadialBasisEncoding(n_rbf=n_rbf)

        # Equivariant GNN layers
        self.gnn_layers = nn.ModuleList(
            [
                TensorProductConvLayer(d_s=d_s, d_v=d_v, n_rbf=n_rbf, hidden_dim=hidden_dim)
                for _ in range(n_layers)
            ]
        )

        # Output heads: predict translation and rotation score vectors
        # These are non-equivariant outputs (just MLPs on aggregated scalar features)
        self.tr_score_head = nn.Sequential(
            nn.Linear(d_s, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.rot_score_head = nn.Sequential(
            nn.Linear(d_s, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        atom_types: torch.Tensor,  # (N,) integer atom type indices
        edge_index: torch.Tensor,  # (2, E) source, destination indices
        edge_attr: torch.Tensor,  # (E, 3) edge vector (dst_xyz - src_xyz)
    ) -> dict:
        """Forward pass over the ligand graph.

        Args:
            atom_types:  (N,) long tensor of atom type indices.
            edge_index:  (2, E) graph connectivity (both directions).
            edge_attr:   (E, 3) edge displacement vectors.

        Returns:
            dict with 'tr_score' (3,) translation score and 'rot_score' (3,) rotation score.
        """
        N = atom_types.shape[0]

        # Node features
        s = self.atom_emb(atom_types)  # (N, d_s)
        # Initialise vector features from scalar embedding (reshape to (N, d_v, 3))
        v_flat = self.v_init(s)  # (N, d_v * 3)
        v = v_flat.view(N, -1, 3)  # (N, d_v, 3)
        # Normalise initial vectors to unit sphere (then layer-norm controls magnitude)
        v = F.normalize(v.view(N, -1), dim=-1).view(N, -1, 3)

        # Edge features: distance and direction
        edge_dist = edge_attr.norm(dim=-1).clamp(min=1e-8)  # (E,)
        edge_dir = edge_attr / edge_dist.unsqueeze(-1)  # (E, 3) unit vectors
        edge_rbf = self.rbf(edge_dist)  # (E, n_rbf)

        # Equivariant GNN layers
        for layer in self.gnn_layers:
            s, v = layer(s, v, edge_index, edge_rbf, edge_dir)

        # Global pooling: mean over all atoms
        s_global = s.mean(dim=0, keepdim=True)  # (1, d_s)

        # Predict scores
        tr_score = self.tr_score_head(s_global).squeeze(0)  # (3,)
        rot_score = self.rot_score_head(s_global).squeeze(0)  # (3,)

        return {"tr_score": tr_score, "rot_score": rot_score}


# ---------------------------------------------------------------------------
# Build functions and example inputs
# ---------------------------------------------------------------------------


def build_diffdock() -> nn.Module:
    """Build a compact DiffDock score model (2 equivariant TP-conv layers, 10 atoms)."""
    return DiffDockScoreModel(
        n_atom_types=9,
        d_s=32,
        d_v=8,
        n_rbf=16,
        n_layers=2,
        hidden_dim=64,
    )


def _make_fully_connected_edges(n: int) -> torch.Tensor:
    """Build a fully-connected edge_index for n nodes."""
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def example_input() -> list:
    """Example (atom_types, edge_index, edge_attr) for DiffDock score model."""
    N = 10  # 10 ligand atoms (small)
    atom_types = torch.randint(0, 9, (N,))
    edge_index = _make_fully_connected_edges(N)  # (2, N*(N-1))
    E = edge_index.shape[1]
    # Random edge displacement vectors (distances 1-5 angstroms)
    edge_attr = torch.randn(E, 3) * 2.0
    return [atom_types, edge_index, edge_attr]


MENAGERIE_ENTRIES = [
    (
        "DiffDock score model (SE(3)-equivariant tensor-product GNN for ligand pose diffusion)",
        "build_diffdock",
        "example_input",
        "2023",
        "DC",
    ),
]
