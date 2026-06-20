"""Equiformer-V2: SE(3)-equivariant graph attention transformer (faithful V3 representative).

Liao, Wood, Das, Smidt et al. (MIT / Meta FAIR).
  - EquiformerV2: arXiv:2306.12059 ("Improved Equivariant Transformer for Scaling to
    Higher-Degree Representations"), official repo ``atomic-architects/equiformer_v2``.
  - Equiformer (V1): arXiv:2206.11990 ("Equivariant Graph Attention Transformer for 3D
    Atomistic Graphs").
  - EquiformerV3: arXiv:2604.09130 ("Scaling Efficient, Expressive, and General
    SE(3)-Equivariant Graph Attention Transformers").

NOTE ON V3 vs V2: the catalog row targets *EquiformerV3*, but the V3-specific deltas
(eSCN tensor-product factorization, SwiGLU-S2 grid activation, equivariant *merged*
layer norm, smooth-cutoff envelope in attention, torch.compile fusions) are software /
efficiency refinements layered on top of the SAME distinctive structural primitive that
V2 introduced and V1 originated: **equivariant graph attention over irreps node features
with spherical-harmonic edge embeddings**.  Because block-level V3 detail beyond those
efficiency tweaks is thin, a faithful compact **Equiformer-V2** is the honest
representative of the family; we additionally fold in two of V3's *structural* ideas that
map cleanly to plain torch -- an equivariant **merged** RMS layer norm (single shared
RMS across all degrees) and a smooth **radial-cutoff envelope** on attention weights --
and note this explicitly.

THE DISTINCTIVE PRIMITIVE (reproduced faithfully, in plain torch, NO e3nn):
Equivariant graph attention messages over irreps node features.  Each node carries
irreps features split into an l=0 *scalar* channel (rotation-invariant) and an l=1
*vector* channel (rotates with the frame).  Edges carry spherical-harmonic embeddings of
their normalized direction (Y0 ~ const, Y1 ~ (x, y, z)).  A tensor-product-style message
mixes scalar and vector features equivariantly using the edge SH; attention weights are
computed from **invariant scalars only** (so the weights themselves are rotation
invariant) and a smooth radial envelope; an equivariant gate/norm updates the features.
Under any global 3D rotation R of the atom positions: scalar channels are unchanged and
vector channels transform as v -> v R^T -- i.e. the whole network is SE(3)-equivariant.

This is a faithful COMPACT random-init reimplementation: tiny hidden sizes, a handful of
atoms, l in {0, 1} only (V2/V3 go to L_max=6+), so the unrolled graph stays renderable.
Random init, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def vector_channel_linear(weight: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Equivariant channel-mixing linear on vector irreps.

    ``v`` is ``(..., C, 3)``; ``weight`` is ``(C_out, C_in)``.  Mixes the CHANNEL dimension
    while leaving the 3 spatial components untouched -- this is the only way a linear map can
    act on l=1 features without breaking equivariance (no bias, no cross-component mixing).
    """
    return torch.einsum("...cd,oc->...od", v, weight)


def spherical_harmonics_l01(edge_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real spherical harmonics for l=0 and l=1 from unit edge directions.

    ``edge_unit`` is ``(E, 3)`` normalized directions. Returns ``(Y0, Y1)`` with
    ``Y0`` shape ``(E, 1)`` (constant, rotation-invariant) and ``Y1`` shape ``(E, 3)``
    (proportional to the direction itself, transforms like a vector under rotation).
    """
    Y0 = edge_unit.new_full((edge_unit.shape[0], 1), 0.282_094_8)  # 1/(2*sqrt(pi))
    Y1 = 0.488_602_5 * edge_unit  # sqrt(3/(4pi)) * (x, y, z) (order y,z,x in e3nn; sign-free here)
    return Y0, Y1


class EquivariantMergedLayerNorm(nn.Module):
    """Equivariant layer norm with a single *merged* RMS across degrees (a V3 idea).

    The l=0 scalar part is normalized as usual; the l=1 vector part is normalized by its
    rotation-INVARIANT norm (so direction is preserved -> equivariant).  V3 shares one RMS
    statistic across all degrees rather than per-degree; we mirror that by computing a
    single merged RMS over the (scalar + per-channel vector-norm) energy.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scalar_weight = nn.Parameter(torch.ones(channels))
        self.vector_weight = nn.Parameter(torch.ones(channels))

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # s: (N, C) scalars ; v: (N, C, 3) vectors
        vnorm = v.norm(dim=-1)  # (N, C) invariant
        merged_energy = s.pow(2).mean(-1, keepdim=True) + vnorm.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(0.5 * merged_energy + self.eps)  # (N, 1) shared across degrees
        s_out = s * inv_rms * self.scalar_weight
        v_out = v * inv_rms.unsqueeze(-1) * self.vector_weight.unsqueeze(-1)
        return s_out, v_out


class EquivariantGate(nn.Module):
    """Gated nonlinearity: SiLU on scalars; vectors gated by an invariant sigmoid scale.

    Applying a nonlinearity directly to vector components would break equivariance, so the
    vector channel is only *scaled* by a rotation-invariant gate derived from the scalars
    (the V1/V2 "gate" trick; V3 generalizes this to its SwiGLU-S2 grid activation).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(channels, channels)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_out = F.silu(s)
        gate = torch.sigmoid(self.gate(s)).unsqueeze(-1)  # (N, C, 1) invariant
        return s_out, v * gate


class EquivariantGraphAttention(nn.Module):
    """Equivariant graph attention -- the distinctive Equiformer primitive.

    For each edge (i -> j): build a tensor-product-style message that mixes the source
    irreps (scalar s_i, vector v_i) with the edge spherical harmonics (Y0, Y1) so the
    result is equivariant.  Attention logits are computed from INVARIANT scalars only
    (so the attention weights are rotation invariant), scaled by a smooth radial-cutoff
    envelope (a V3 idea), then softmax-normalized over neighbors.
    """

    def __init__(self, channels: int, n_head: int = 2, cutoff: float = 3.0) -> None:
        super().__init__()
        self.c = channels
        self.h = n_head
        self.hd = channels // n_head
        self.cutoff = cutoff
        # scalar (l=0) projections
        self.src_scalar = nn.Linear(channels, channels)
        self.dst_scalar = nn.Linear(channels, channels)
        # edge-conditioned scalar weighting (depends on invariant edge length embedding)
        self.edge_scalar = nn.Linear(1, channels)
        # tensor-product mixers (scalar<->vector via edge SH), equivariant by construction
        self.tp_s_from_v = nn.Linear(channels, channels, bias=False)  # v.Y1 -> scalar
        self.tp_v_from_s = nn.Linear(channels, channels, bias=False)  # s*Y1 -> vector
        self.tp_v_keep = nn.Linear(channels, channels, bias=False)  # carry source vector
        # attention logits from invariant scalars
        self.to_logit = nn.Linear(channels, n_head)
        # output equivariant projections
        self.out_scalar = nn.Linear(channels, channels)
        self.out_vector = nn.Linear(channels, channels, bias=False)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_unit: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]  # (E,)
        Y0, Y1 = spherical_harmonics_l01(edge_unit)  # (E,1), (E,3)

        s_i = self.src_scalar(s)[src]  # (E, C) source scalars
        s_j = self.dst_scalar(s)[dst]  # (E, C) dest scalars
        v_i = v[src]  # (E, C, 3) source vectors

        # invariant edge length embedding (radial), used for both logits and gating
        edge_emb = self.edge_scalar(edge_len.unsqueeze(-1))  # (E, C)

        # --- equivariant tensor-product message ---
        # scalar output: contract source vector with edge SH-1 (invariant scalar = v . Y1)
        v_dot_Y1 = (v_i * Y1.unsqueeze(1)).sum(-1)  # (E, C) invariant
        msg_scalar = self.tp_s_from_v(v_dot_Y1) + (s_i + s_j) * Y0  # (E, C)
        msg_scalar = msg_scalar * edge_emb
        # vector output: lift source scalar onto edge SH-1 direction + carry source vector
        msg_vector = (
            self.tp_v_from_s(s_i).unsqueeze(-1) * Y1.unsqueeze(1)  # (E, C, 3)
            + vector_channel_linear(self.tp_v_keep.weight, v_i)
        )

        # --- attention weights from INVARIANT scalars + smooth radial envelope ---
        logits = self.to_logit(msg_scalar)  # (E, n_head) invariant
        envelope = torch.clamp(1.0 - (edge_len / self.cutoff), min=0.0).pow(2)  # (E,)
        logits = logits + torch.log(envelope.unsqueeze(-1) + 1e-9)

        n_nodes = s.shape[0]
        # softmax over incoming edges per destination node (segment softmax)
        logits = logits - _segment_max(logits, dst, n_nodes)[dst]
        weights = logits.exp()
        denom = _segment_sum(weights, dst, n_nodes)[dst] + 1e-9
        alpha = (weights / denom).unsqueeze(-1)  # (E, n_head, 1)

        # weight the equivariant messages by (invariant) attention
        ms = msg_scalar.view(-1, self.h, self.hd) * alpha
        mv = msg_vector.view(-1, self.h, self.hd, 3) * alpha.unsqueeze(-1)
        ms = ms.reshape(-1, self.c)
        mv = mv.reshape(-1, self.c, 3)

        agg_s = _segment_sum(ms, dst, n_nodes)
        agg_v = _segment_sum(mv.reshape(mv.shape[0], -1), dst, n_nodes).view(n_nodes, self.c, 3)
        return self.out_scalar(agg_s), vector_channel_linear(self.out_vector.weight, agg_v)


def _segment_sum(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = x.new_zeros((n,) + x.shape[1:])
    idx = index.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    return out.scatter_add(0, idx, x)


def _segment_max(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = x.new_full((n,) + x.shape[1:], float("-inf"))
    idx = index.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    return out.scatter_reduce(0, idx, x, reduce="amax", include_self=True)


class EquivariantFeedForward(nn.Module):
    """Equivariant feed-forward: scalar MLP + invariant-gated vector mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = EquivariantMergedLayerNorm(channels)
        self.gate = EquivariantGate(channels)
        self.s1 = nn.Linear(channels, channels * 4)  # V3: 4x FFN hidden
        self.s2 = nn.Linear(channels * 4, channels)
        self.v1 = nn.Linear(channels, channels, bias=False)
        self.v2 = nn.Linear(channels, channels, bias=False)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_n, v_n = self.norm(s, v)
        s_g, v_g = self.gate(s_n, v_n)
        s_out = self.s2(F.silu(self.s1(s_g)))
        v_mix = vector_channel_linear(self.v1.weight, v_g)
        v_out = vector_channel_linear(self.v2.weight, v_mix)
        return s_out, v_out


class EquiformerBlock(nn.Module):
    """One Equiformer Transformer block: equivariant attention + equivariant FFN (residual)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = EquivariantMergedLayerNorm(channels)
        self.attn = EquivariantGraphAttention(channels)
        self.ff = EquivariantFeedForward(channels)

    def forward(self, s, v, edge_index, edge_unit, edge_len):
        s_n, v_n = self.norm(s, v)
        ds, dv = self.attn(s_n, v_n, edge_index, edge_unit, edge_len)
        s = s + ds
        v = v + dv
        ds2, dv2 = self.ff(s, v)
        s = s + ds2
        v = v + dv2
        return s, v


class EquiformerV2(nn.Module):
    """Compact Equiformer-V2 over a small atomic graph.

    Atom embedding -> irreps node features (scalar s + vector v, v init = 0 so that the
    network is exactly equivariant at init) -> N equivariant graph-attention blocks ->
    invariant scalar energy head + equivariant per-atom force-like vector head.
    """

    def __init__(self, n_species: int = 8, channels: int = 16, n_blocks: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.atom_embed = nn.Embedding(n_species, channels)  # l=0 scalar init
        self.blocks = nn.ModuleList(EquiformerBlock(channels) for _ in range(n_blocks))
        self.final_norm = EquivariantMergedLayerNorm(channels)
        self.energy_head = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1)
        )
        self.force_head = nn.Linear(channels, 1, bias=False)  # invariant scale on vector channel

    def forward(self, atom_types: torch.Tensor, positions: torch.Tensor):
        # atom_types: (N,) long ; positions: (N, 3) float
        n = atom_types.shape[0]
        s = self.atom_embed(atom_types)  # (N, C) scalars
        v = positions.new_zeros(n, self.channels, 3)  # (N, C, 3) vectors, equivariant zero init

        # build a dense (all-pairs, no self) edge graph from positions
        idx = torch.arange(n, device=positions.device)
        src = idx.repeat_interleave(n)
        dst = idx.repeat(n)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        edge_index = torch.stack([src, dst], dim=0)
        rel = positions[dst] - positions[src]  # (E, 3)
        edge_len = rel.norm(dim=-1)  # (E,) invariant
        edge_unit = rel / (edge_len.unsqueeze(-1) + 1e-9)  # (E, 3) covariant

        for blk in self.blocks:
            s, v = blk(s, v, edge_index, edge_unit, edge_len)

        s, v = self.final_norm(s, v)
        per_atom_energy = self.energy_head(s)  # (N, 1) invariant
        total_energy = per_atom_energy.sum(0, keepdim=True)  # (1, 1) invariant
        # equivariant per-atom vector ("forces"): invariant-weighted sum of vector channels
        forces = (v * self.force_head.weight.view(1, -1, 1)).sum(1)  # (N, 3) covariant
        return total_energy, forces


def build() -> nn.Module:
    return EquiformerV2()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny molecule: 6 atoms with random species and random 3D positions.

    Returns ``(atom_types (6,), positions (6, 3))`` -- splat into ``tl.trace(m, *x)``.
    """
    torch.manual_seed(0)
    atom_types = torch.randint(0, 8, (6,))
    positions = torch.randn(6, 3)
    return atom_types, positions


MENAGERIE_ENTRIES = [
    (
        "Equiformer-V2 (SE(3)-equivariant graph attention; faithful Equiformer-V3 representative)",
        "build",
        "example_input",
        "2023",
        "DC",
    ),
]
