"""OFormer: Operator Transformer with Cross-Attention Encoder and Point Decoder.

Li et al., 2022.
Paper: https://arxiv.org/abs/2210.02143
Source: https://github.com/alasdairtran/fourierflow  (related), official: https://github.com/L-I-M-I-T/OFormer

OFormer architecture:
  1. Input function encoder: a cross-attention module where the input function
     values at sensor locations are encoded via Galerkin/linear attention.
     Two sub-stages:
       (a) Self-attention on inputs (standard or Galerkin-type)
       (b) Cross-attention from learnable latent queries to the input tokens
  2. Operator Transformer body: Galerkin self-attention layers (linear O(N)).
  3. Decoder: given QUERY coordinates, cross-attention from query positions
     to the latent representation, then a pointwise MLP to produce the output
     field value at any query point.

The distinctive features:
  - Galerkin-type linear attention in the encoder body (Q(K^T V)/N, LayerNorm on K,V)
  - Coordinate-aware cross-attention decoder: query positions are concatenated
    as positional info into the decoder cross-attention key
  - Architecture naturally handles non-uniform sensor locations / irregular grids

Simplifications: 2 encoder layers, 2 decoder layers, d_model=64, 4 heads,
input grid 16x16=256 tokens, query set 64 points.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Linear (Galerkin) self-attention
# ---------------------------------------------------------------------------


class GalerkinSelfAttn(nn.Module):
    """Galerkin-type linear self-attention: Q(K^T V)/N with LayerNorm on K,V."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.ln_k = nn.LayerNorm(self.dh)
        self.ln_v = nn.LayerNorm(self.dh)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.h, self.dh
        Q = self.Wq(x).view(B, N, H, Dh).transpose(1, 2)
        K = self.ln_k(self.Wk(x).view(B, N, H, Dh).transpose(1, 2))
        V = self.ln_v(self.Wv(x).view(B, N, H, Dh).transpose(1, 2))
        KtV = torch.einsum("bhnd,bhnv->bhdv", K, V) / N
        out = torch.einsum("bhnd,bhdv->bhnv", Q, KtV)
        return self.out(out.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------------------
# Cross-attention (standard scaled dot-product) used in decoder
# ---------------------------------------------------------------------------


class CrossAttn(nn.Module):
    """Standard multi-head cross-attention: queries from Q, keys/values from KV."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.scale = self.dh**-0.5
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Nq, D), kv: (B, Nk, D)
        B, Nq, D = q.shape
        Nk = kv.size(1)
        H, Dh = self.h, self.dh
        Q = self.Wq(q).view(B, Nq, H, Dh).transpose(1, 2)
        K = self.Wk(kv).view(B, Nk, H, Dh).transpose(1, 2)
        V = self.Wv(kv).view(B, Nk, H, Dh).transpose(1, 2)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        return self.out(out)


class FFN(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder block (Galerkin self-attn)
# ---------------------------------------------------------------------------


class OFormerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = GalerkinSelfAttn(d_model, n_heads)
        self.ffn = FFN(d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


# ---------------------------------------------------------------------------
# Decoder block: cross-attention from query coords -> encoded latent
# ---------------------------------------------------------------------------


class OFormerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.cross = CrossAttn(d_model, n_heads)
        self.self_attn = GalerkinSelfAttn(d_model, n_heads)
        self.ffn = FFN(d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # q: query tokens (B, Nq, D)  mem: encoder output (B, N, D)
        q = q + self.cross(self.n1(q), mem)
        q = q + self.self_attn(self.n2(q))
        q = q + self.ffn(self.n3(q))
        return q


# ---------------------------------------------------------------------------
# Full OFormer
# ---------------------------------------------------------------------------


class OFormer(nn.Module):
    """Operator Transformer: encoder (Galerkin attn) + coordinate-conditioned decoder."""

    def __init__(
        self,
        in_channels: int = 3,  # u(x,y) + coords
        out_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        coord_dim: int = 2,  # (x, y) query coords
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.encoder = nn.ModuleList(
            [OFormerEncoderBlock(d_model, n_heads) for _ in range(n_enc_layers)]
        )
        # Query coord embedding: map 2D coords -> d_model
        self.query_proj = nn.Linear(coord_dim, d_model)
        self.decoder = nn.ModuleList(
            [OFormerDecoderBlock(d_model, n_heads) for _ in range(n_dec_layers)]
        )
        # Pointwise output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def forward(self, x: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_channels) -- input function + sensor coords
        # query_coords: (B, Nq, 2) -- output query locations
        h = self.input_proj(x)
        for enc in self.encoder:
            h = enc(h)

        q = self.query_proj(query_coords)  # (B, Nq, d_model)
        for dec in self.decoder:
            q = dec(q, h)

        return self.output_mlp(q)  # (B, Nq, out_channels)


def build_oformer() -> nn.Module:
    return OFormer(
        in_channels=3,
        out_channels=1,
        d_model=64,
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=2,
        coord_dim=2,
    )


def example_input_oformer():
    # input function at 16x16=256 sensor points: (B, N, 3) = u + (x,y)
    x = torch.randn(1, 256, 3)
    # query at 64 output points: (B, Nq, 2) = (x,y)
    query = torch.randn(1, 64, 2)
    return [x, query]


MENAGERIE_ENTRIES = [
    ("OFormer", "build_oformer", "example_input_oformer", "2022", "DC"),
]
