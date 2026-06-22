"""GNOT: General Neural Operator Transformer.

Hao et al., 2023.
Paper: https://arxiv.org/abs/2302.14376
Source: https://github.com/HaoZhongkai/GNOT

GNOT is a neural operator transformer designed to handle heterogeneous inputs
(multiple functions defined on different grids/point clouds) and complex geometries.

Key distinctive primitives:
  1. Heterogeneous Normalized Cross-Attention (HNCA): cross-attention from query
     points to input function tokens, with per-input-type normalization. Handles
     N different input functions {u1, u2, ...} each defined on its own point set.
  2. Geometric Gating: attention weights are modulated by a learned geometric
     compatibility gate using input coordinate differences.
  3. Mixture-of-Experts (MoE) FFN: the feedforward block uses a top-k MoE where
     each "expert" is a small MLP and the router is a softmax gate.
  4. Input function aggregation: multiple input function tokens are concatenated
     along the sequence dimension before cross-attention.

Architecture: Transformer with HNCA layers + MoE-FFN. Query is the output grid;
keys/values come from input function(s) on their respective point sets.

Simplifications: 2 HNCA+MoE layers, d_model=64, 4 heads, 2 input functions on
8 points each, 32 query output points, 2 MoE experts (top-1 routing). Geometric
gating uses a small MLP on ||q_coord - k_coord||^2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricGate(nn.Module):
    """Geometric gating: MLP on squared coordinate distance -> scalar gate."""

    def __init__(self, coord_dim: int = 2, d_model: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, q_coords: torch.Tensor, k_coords: torch.Tensor) -> torch.Tensor:
        # q_coords: (B, Nq, 2), k_coords: (B, Nk, 2)
        # returns: (B, Nq, Nk) gate weights
        diff = q_coords.unsqueeze(2) - k_coords.unsqueeze(1)  # (B,Nq,Nk,2)
        dist2 = (diff**2).sum(-1, keepdim=True)  # (B,Nq,Nk,1)
        gate = self.mlp(dist2).squeeze(-1)  # (B,Nq,Nk)
        return gate


class HeterogeneousCrossAttention(nn.Module):
    """Heterogeneous Normalized Cross-Attention (HNCA).

    Query from output grid; Keys/Values from concatenated input functions.
    Geometric gating modulates the attention scores.
    """

    def __init__(self, d_model: int, n_heads: int, coord_dim: int = 2) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = self.dh**-0.5
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.geo_gate = GeometricGate(coord_dim, d_model)
        # Per-input-type layer normalization (simplified: one shared norm)
        self.k_norm = nn.LayerNorm(d_model)
        self.v_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,  # (B, Nq, d_model)
        kv_tokens: torch.Tensor,  # (B, Nk, d_model) -- concatenated input functions
        q_coords: torch.Tensor,  # (B, Nq, 2)
        k_coords: torch.Tensor,  # (B, Nk, 2)
    ) -> torch.Tensor:
        B, Nq, D = query.shape
        Nk = kv_tokens.size(1)
        H, Dh = self.h, self.dh

        Q = self.Wq(query).view(B, Nq, H, Dh).transpose(1, 2)
        K = self.Wk(self.k_norm(kv_tokens)).view(B, Nk, H, Dh).transpose(1, 2)
        V = self.Wv(self.v_norm(kv_tokens)).view(B, Nk, H, Dh).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = Q @ K.transpose(-2, -1) * self.scale  # (B, H, Nq, Nk)

        # Geometric gating: (B, Nq, Nk) -> broadcast over heads
        geo = self.geo_gate(q_coords, k_coords).unsqueeze(1)  # (B, 1, Nq, Nk)
        scores = scores * geo

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        return self.out(out)


class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN: top-k routing over n_experts small MLPs."""

    def __init__(self, d_model: int, n_experts: int = 2, top_k: int = 1) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # Router
        self.router = nn.Linear(d_model, n_experts)
        # Experts: each a small 2-layer MLP
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        B, N, D = x.shape
        router_logits = self.router(x)  # (B, N, n_experts)
        top_weights, top_idx = torch.topk(
            torch.softmax(router_logits, dim=-1), self.top_k, dim=-1
        )  # (B, N, top_k)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = top_idx[..., k]  # (B, N)
            w = top_weights[..., k : k + 1]  # (B, N, 1)
            for e in range(self.n_experts):
                mask = (idx == e).float().unsqueeze(-1)  # (B, N, 1)
                expert_out = self.experts[e](x)
                out = out + mask * w * expert_out
        return out


class GNOTBlock(nn.Module):
    """One GNOT block: HNCA + MoE-FFN."""

    def __init__(self, d_model: int, n_heads: int, n_experts: int = 2) -> None:
        super().__init__()
        self.hnca = HeterogeneousCrossAttention(d_model, n_heads)
        self.moe_ffn = MoEFFN(d_model, n_experts)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv_tokens: torch.Tensor,
        q_coords: torch.Tensor,
        k_coords: torch.Tensor,
    ) -> torch.Tensor:
        query = query + self.hnca(self.n1(query), kv_tokens, q_coords, k_coords)
        query = query + self.moe_ffn(self.n2(query))
        return query


class GNOT(nn.Module):
    """General Neural Operator Transformer.

    Handles multiple heterogeneous input functions on different point clouds.
    Output: function values at query positions.
    """

    def __init__(
        self,
        n_input_funcs: int = 2,  # number of heterogeneous input functions
        in_channels: int = 1,  # channels per input function
        coord_dim: int = 2,  # spatial coordinate dimension
        out_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_experts: int = 2,
    ) -> None:
        super().__init__()
        # Project each input function to d_model (shared projection; distinct bias per type)
        self.input_proj = nn.ModuleList(
            [nn.Linear(in_channels + coord_dim, d_model) for _ in range(n_input_funcs)]
        )
        # Query point projection: coords -> d_model
        self.query_proj = nn.Linear(coord_dim + out_channels, d_model)

        self.blocks = nn.ModuleList(
            [GNOTBlock(d_model, n_heads, n_experts) for _ in range(n_layers)]
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def forward(
        self,
        input_funcs: list,  # list of (B, Ni, in_channels) tensors
        input_coords: list,  # list of (B, Ni, coord_dim) tensors
        query_coords: torch.Tensor,  # (B, Nq, coord_dim)
    ) -> torch.Tensor:
        # Project each input function (concatenate with its coordinates)
        kv_list = []
        k_coord_list = []
        for i, (u, c) in enumerate(zip(input_funcs, input_coords)):
            uc = torch.cat([u, c], dim=-1)
            kv_list.append(self.input_proj[i](uc))
            k_coord_list.append(c)

        # Concatenate all input function tokens
        kv_tokens = torch.cat(kv_list, dim=1)  # (B, sum(Ni), d_model)
        k_coords = torch.cat(k_coord_list, dim=1)  # (B, sum(Ni), coord_dim)

        # Query: zero init output + coordinates
        B, Nq, cd = query_coords.shape
        query_in = torch.cat(
            [query_coords, torch.zeros(B, Nq, 1, device=query_coords.device)], dim=-1
        )
        query = self.query_proj(query_in)  # (B, Nq, d_model)

        for blk in self.blocks:
            query = blk(query, kv_tokens, query_coords, k_coords)

        return self.output_mlp(query)  # (B, Nq, out_channels)


def build_gnot() -> nn.Module:
    return GNOT(
        n_input_funcs=2,
        in_channels=1,
        coord_dim=2,
        out_channels=1,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_experts=2,
    )


def example_input_gnot():
    # 2 input functions, 8 points each, on 2D grids; 32 query output points
    u1 = torch.randn(1, 8, 1)
    c1 = torch.rand(1, 8, 2)
    u2 = torch.randn(1, 8, 1)
    c2 = torch.rand(1, 8, 2)
    q_coords = torch.rand(1, 32, 2)
    return [[u1, u2], [c1, c2], q_coords]


MENAGERIE_ENTRIES = [
    (
        "GNOT (General Neural Operator Transformer)",
        "build_gnot",
        "example_input_gnot",
        "2023",
        "DC",
    ),
]
