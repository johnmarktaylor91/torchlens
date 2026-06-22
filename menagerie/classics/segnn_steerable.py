"""SEGNN: Steerable E(3) Equivariant Graph Neural Network.

Brandstetter et al., "Geometric and Physical Quantities Improve E(3)
Equivariant Message Passing." arXiv:2110.02905 (ICLR 2022).
Source: https://github.com/RobDHess/Steerable-E3-GNN

Distinctive primitive:
  Steerable equivariant message passing where messages and node updates
  are computed by "steerable MLPs" -- operations that mix features within
  irrep types and use Clebsch-Gordan (CG) tensor products conditioned on
  geometric (steerable) edge attributes embedded as spherical harmonics.

  The architecture:
    (1) Node features: type-0 (scalars) + type-1 (3-vectors).
    (2) Edge attributes: SH embeddings of normalized edge direction (l=0,1).
    (3) Message: CG-mix of node features conditioned on edge attrs.
    (4) Aggregation: sum over neighbors.
    (5) Node update: CG-mix of aggregated message conditioned on identity attr.

  CG paths for l_max=1 that we implement (real, non-zero paths):
    0 x 0 -> 0   (scalar * scalar -> scalar)
    0 x 1 -> 1   (scalar * vector -> vector)
    1 x 0 -> 1   (vector * scalar -> vector)
    1 x 1 -> 0   (vector dot vector -> scalar)

Faithful-compact simplifications:
  - l_max = 1 (type-0 scalars + type-1 vectors).
  - Hidden channels: c0=8 scalars, c1=4 vectors.
  - 2 message-passing layers, 5-node fully-connected graph.
  - CG operations implemented explicitly in plain torch.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Spherical harmonics l=0 and l=1
# =============================================================================


def sh_l0(unit: torch.Tensor) -> torch.Tensor:
    """Y_0^0 constant: (E, 1)"""
    return unit.new_ones(unit.size(0), 1) * (1.0 / math.sqrt(4 * math.pi))


def sh_l1(unit: torch.Tensor) -> torch.Tensor:
    """Y_1^m ~ (y, z, x): (E, 3)"""
    c = math.sqrt(3.0 / (4.0 * math.pi))
    return c * torch.stack([unit[:, 1], unit[:, 2], unit[:, 0]], dim=-1)


# =============================================================================
# CG tensor product layer (type-0, type-1 features x type-0, type-1 attr)
# =============================================================================


class CG01Block(nn.Module):
    """Steerable MLP block using l<=1 CG paths.

    Inputs:
      feat_s: (..., c0_in)   type-0 features (scalars)
      feat_v: (..., c1_in, 3) type-1 features (vectors)
      attr_s: (..., a0)       type-0 edge attr (scalar SH)
      attr_v: (..., a1, 3)    type-1 edge attr (vector SH)

    CG paths:
      0x0->0: feat_s * attr_s -> new scalars
      1x1->0: dot(feat_v, attr_v) -> new scalars
      0x1->1: feat_s * attr_v -> new vectors
      1x0->1: feat_v * attr_s -> new vectors
    Then linear mixing to c0_out scalars, c1_out vectors.
    """

    def __init__(self, c0_in: int, c1_in: int, a0: int, a1: int, c0_out: int, c1_out: int) -> None:
        super().__init__()
        # Number of scalar paths: c0_in*a0 (0x0->0) + c1_in*a1 (1x1->0)
        n_s = c0_in * a0 + c1_in * a1
        # Number of vector paths: c0_in*a1 (0x1->1) + c1_in*a0 (1x0->1)
        n_v = c0_in * a1 + c1_in * a0
        self.w_s = nn.Linear(max(n_s, 1), c0_out, bias=True)
        self.w_v = nn.Linear(max(n_v, 1), c1_out, bias=False)
        self.c0_in = c0_in
        self.c1_in = c1_in
        self.a0 = a0
        self.a1 = a1
        self.n_s = n_s
        self.n_v = n_v

    def forward(
        self,
        feat_s: torch.Tensor,  # (..., c0_in)
        feat_v: torch.Tensor,  # (..., c1_in, 3)
        attr_s: torch.Tensor,  # (..., a0)
        attr_v: torch.Tensor,  # (..., a1, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts_s = []
        parts_v = []

        # 0x0->0
        if self.c0_in > 0 and self.a0 > 0:
            # (..., c0_in) x (..., a0) -> (..., c0_in*a0)
            p = feat_s.unsqueeze(-1) * attr_s.unsqueeze(-2)  # (..., c0_in, a0)
            parts_s.append(p.flatten(-2))

        # 1x1->0: dot product
        if self.c1_in > 0 and self.a1 > 0:
            # (..., c1_in, 3) x (..., a1, 3) -> (..., c1_in, a1)
            d = torch.einsum("...id,...jd->...ij", feat_v, attr_v)
            parts_s.append(d.flatten(-2))

        # 0x1->1
        if self.c0_in > 0 and self.a1 > 0:
            # (..., c0_in) x (..., a1, 3) -> (..., c0_in*a1, 3)
            p = feat_s.unsqueeze(-1).unsqueeze(-1) * attr_v.unsqueeze(-3)  # (...,c0,a1,3)
            parts_v.append(p.flatten(-3, -2))

        # 1x0->1
        if self.c1_in > 0 and self.a0 > 0:
            # (..., c1_in, 3) x (..., a0) -> (..., c1_in*a0, 3)
            p = feat_v.unsqueeze(-2) * attr_s.unsqueeze(-1).unsqueeze(-3)  # (...,c1,a0,3)
            parts_v.append(p.flatten(-3, -2))

        if parts_s:
            inp_s = torch.cat(parts_s, dim=-1)
        else:
            inp_s = feat_s.new_zeros(*feat_s.shape[:-1], 1)

        if parts_v:
            inp_v = torch.cat(parts_v, dim=-2)  # (..., n_v, 3)
        else:
            inp_v = feat_v.new_zeros(*feat_v.shape[:-2], 1, 3)

        new_s = self.w_s(inp_s)  # (..., c0_out)
        # w_v maps channel dim (n_v -> c1_out), broadcast over spatial 3
        new_v = torch.einsum("oc,...id->...od", self.w_v.weight, inp_v)  # (..., c1_out, 3)
        return new_s, new_v


class SEGNNLayer(nn.Module):
    """One SEGNN message-passing layer."""

    def __init__(self, c0: int, c1: int) -> None:
        super().__init__()
        # Edge attrs: a0=1 (l=0 SH), a1=3 (l=1 SH as 1 vector = a1=1 type-1)
        # We treat l=1 SH as 1 type-1 feature (3-vector), so a1=1
        a0, a1 = 1, 1  # one type-0 and one type-1 edge attribute
        # Message: input = [feat_i, feat_j] -> 2*c0 scalars, 2*c1 vectors
        self.message_cg = CG01Block(2 * c0, 2 * c1, a0, a1, c0, c1)
        # Update: input = [agg_s + h_s, agg_v + h_v]
        self.update_cg = CG01Block(2 * c0, 2 * c1, a0, a1, c0, c1)
        self.norm_s = nn.LayerNorm(c0)
        self.c0 = c0
        self.c1 = c1

    def forward(
        self,
        s: torch.Tensor,  # (N, c0)
        v: torch.Tensor,  # (N, c1, 3)
        pos: torch.Tensor,  # (N, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = s.size(0)

        # Edge attributes (SH of normalized direction)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        unit = diff / dist
        # type-0 attr: scalar (N, N, 1)
        a0_edge = sh_l0(unit.view(N * N, 3)).view(N, N, 1)
        # type-1 attr: 1 vector of 3-components -> (N, N, 1, 3)
        a1_edge = sh_l1(unit.view(N * N, 3)).view(N, N, 3).unsqueeze(-2)  # (N,N,1,3)

        # Pairwise features
        s_i = s.unsqueeze(1).expand(N, N, -1)
        s_j = s.unsqueeze(0).expand(N, N, -1)
        v_i = v.unsqueeze(1).expand(N, N, -1, -1)
        v_j = v.unsqueeze(0).expand(N, N, -1, -1)
        pair_s = torch.cat([s_i, s_j], dim=-1)  # (N, N, 2*c0)
        pair_v = torch.cat([v_i, v_j], dim=-2)  # (N, N, 2*c1, 3)

        # Message
        msg_s, msg_v = self.message_cg(pair_s, pair_v, a0_edge, a1_edge)
        # (N, N, c0), (N, N, c1, 3)

        # Zero self-edges
        eye = torch.eye(N, device=s.device)
        msg_s = msg_s * (1 - eye).unsqueeze(-1)
        msg_v = msg_v * (1 - eye).unsqueeze(-1).unsqueeze(-1)

        agg_s = msg_s.sum(dim=1)  # (N, c0)
        agg_v = msg_v.sum(dim=1)  # (N, c1, 3)

        # Update: concat [h, agg]
        upd_s = torch.cat([s, agg_s], dim=-1)  # (N, 2*c0)
        upd_v = torch.cat([v, agg_v], dim=-2)  # (N, 2*c1, 3)
        # Node self-attr: trivial (constant type-0=1, zero type-1)
        self_a0 = s.new_ones(N, 1) / math.sqrt(4 * math.pi)
        self_a1 = s.new_zeros(N, 1, 3)

        new_s, new_v = self.update_cg(upd_s, upd_v, self_a0, self_a1)
        new_s = F.silu(self.norm_s(new_s))
        # Equivariant activation on vectors: scale by silu of norm
        vnorm = new_v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        new_v = new_v * F.silu(vnorm) / vnorm
        return new_s, new_v


class SEGNN(nn.Module):
    """Steerable E(3) Graph Neural Network."""

    def __init__(
        self, c0_in: int = 1, c0: int = 8, c1: int = 4, n_layers: int = 2, n_classes: int = 2
    ) -> None:
        super().__init__()
        self.embed_s = nn.Linear(c0_in, c0)
        self.c1 = c1
        # Initial vector embed from positions (type-1 equivariant)
        self.embed_v_scale = nn.Parameter(torch.ones(c1))
        self.layers = nn.ModuleList([SEGNNLayer(c0, c1) for _ in range(n_layers)])
        self.cls = nn.Sequential(nn.Linear(c0, c0), nn.SiLU(), nn.Linear(c0, n_classes))

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: (N, c0_in),  pos: (N, 3)  ->  logits (n_classes,)"""
        N = pos.size(0)
        s = F.silu(self.embed_s(x))  # (N, c0)
        # Init type-1 features: c1 copies of normalized position (equivariant init)
        pos_n = F.normalize(pos, dim=-1)  # (N, 3)
        v = pos_n.unsqueeze(1) * self.embed_v_scale.view(1, self.c1, 1)  # (N, c1, 3)
        for layer in self.layers:
            s, v = layer(s, v, pos)
        return self.cls(s.mean(dim=0))


def build_segnn() -> nn.Module:
    return SEGNN(c0_in=1, c0=8, c1=4, n_layers=2, n_classes=2)


def example_input_segnn() -> list[torch.Tensor]:
    torch.manual_seed(2)
    x = torch.randn(5, 1)
    pos = torch.randn(5, 3)
    return [x, pos]


MENAGERIE_ENTRIES = [
    (
        "SEGNN (Steerable E(3) GNN)",
        "build_segnn",
        "example_input_segnn",
        "2022",
        "DC",
    ),
]
