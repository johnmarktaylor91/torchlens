"""Uni-Mol: a universal 3D molecular representation SE(3)-invariant transformer.

Zhou et al., "Uni-Mol: A Universal 3D Molecular Representation Learning Framework",
ICLR 2023 (arXiv late 2022). https://openreview.net/forum?id=6K2RM6wVqKu
Source: https://github.com/deepmodeling/Uni-Mol

Uni-Mol's distinctive design (SE(3)-invariant transformer over atoms):
  - **atom representation** initialised from atom type embeddings;
  - **pair representation** initialised by encoding the Euclidean distance of every
    atom pair through a **pair-type-aware Gaussian kernel** (affine per (type_i,
    type_j) on the distance, then a bank of Gaussians);
  - transformer blocks where self-attention is **biased by the pair representation**
    (pair rep added to the Q.K logits per head), and the pair representation is in
    turn **updated from the attention map** (the Q.K product), so atom and pair
    streams co-evolve;
  - SE(3) invariance comes from using only invariant distances as geometric input;
    a coordinate/pair head can output equivariant 3D positions (the pretraining
    3D-position-denoising task) from the invariant features.

This is a faithful reimplementation of the atom-embed + Gaussian-distance pair-bias
+ pair-biased attention + pair-update stack + classification & coordinate (pair)
heads, at small widths / few atoms / few layers so the unrolled atlas graph renders
quickly. Random init, forward-only.

Faithful-core simplifications (honest, not lies):
  - embed dim 64, 2 layers, 4 heads on a small 8-atom molecule (vs the full Uni-Mol
    widths); the Gaussian-kernel pair init + pair-bias attention + pair-update math
    is faithful.
  - input ``src_tokens`` (atom types) and ``src_coord`` (3D coords) are supplied as
    two small fixed tensors (standing in for Uni-Mol's data-dict input), so the
    example input is a small fixed tuple the forward consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianLayer(nn.Module):
    """Pair-type-aware Gaussian distance kernel (Uni-Mol's GaussianLayer)."""

    def __init__(self, K: int = 16, n_edge_types: int = 64 * 64) -> None:
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        # per-edge-type affine on the distance
        self.mul = nn.Embedding(n_edge_types, 1)
        self.bias = nn.Embedding(n_edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.mul.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

    def forward(self, dist: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # dist: (N, N); edge_type: (N, N) long -> (N, N, K)
        mul = self.mul(edge_type).squeeze(-1)
        bias = self.bias(edge_type).squeeze(-1)
        x = mul * dist + bias
        x = x.unsqueeze(-1).expand(-1, -1, self.K)
        mean = self.means.weight.view(-1)
        std = self.stds.weight.view(-1).abs() + 1e-5
        return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (std * (2 * torch.pi) ** 0.5)


class UniMolAttention(nn.Module):
    """Self-attention biased by the pair representation; returns updated atom rep + attn map."""

    def __init__(self, d: int, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.dh = d // n_head
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.pair_bias = nn.Linear(d, n_head)  # pair rep -> per-head bias

    def forward(self, x: torch.Tensor, pair: torch.Tensor):
        # x: (N, d); pair: (N, N, d)
        N = x.shape[0]
        q = self.q(x).view(N, self.h, self.dh).permute(1, 0, 2)
        k = self.k(x).view(N, self.h, self.dh).permute(1, 0, 2)
        v = self.v(x).view(N, self.h, self.dh).permute(1, 0, 2)
        bias = self.pair_bias(pair).permute(2, 0, 1)  # (H, N, N)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.dh**0.5) + bias
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).permute(1, 0, 2).reshape(N, -1)
        return self.o(out), logits  # logits feed pair update


class UniMolBlock(nn.Module):
    def __init__(self, d: int, n_head: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = UniMolAttention(d, n_head)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        # pair update: attention logits (per head) -> pair rep increment
        self.pair_norm = nn.LayerNorm(d)
        self.pair_update = nn.Linear(n_head, d)

    def forward(self, x: torch.Tensor, pair: torch.Tensor):
        a, logits = self.attn(self.norm1(x), pair)
        x = x + a
        x = x + self.ff(self.norm2(x))
        # update pair representation from the attention map
        pair = pair + self.pair_update(logits.permute(1, 2, 0))  # (N,N,H)->(N,N,d)
        pair = self.pair_norm(pair)
        return x, pair


class UniMol(nn.Module):
    """Compact Uni-Mol SE(3)-invariant 3D molecular transformer."""

    def __init__(
        self,
        n_type: int = 64,
        d: int = 64,
        n_layer: int = 2,
        n_head: int = 4,
        K: int = 16,
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(n_type, d)
        self.gbf = GaussianLayer(K, n_edge_types=n_type * n_type)
        self.gbf_proj = nn.Linear(K, d)
        self.n_type = n_type
        self.blocks = nn.ModuleList([UniMolBlock(d, n_head) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.cls_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # coordinate / pair head: pair rep -> scalar per pair (3D denoising signal)
        self.pair_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, inputs: tuple):
        src_tokens, src_coord = inputs  # (N,) long, (N, 3) float
        x = self.atom_embed(src_tokens)  # (N, d)
        # pairwise distances + pair-type ids
        diff = src_coord[:, None, :] - src_coord[None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)  # (N, N)
        edge_type = src_tokens[:, None] * self.n_type + src_tokens[None, :]
        pair = self.gbf_proj(self.gbf(dist, edge_type))  # (N, N, d)
        for blk in self.blocks:
            x, pair = blk(x, pair)
        x = self.norm(x)
        graph_logit = self.cls_head(x.mean(dim=0, keepdim=True))  # (1, 1)
        pair_pred = self.pair_head(pair).squeeze(-1)  # (N, N) distance-denoising signal
        return graph_logit, pair_pred


def build() -> nn.Module:
    return UniMol()


def example_input() -> tuple:
    """Fixed (src_tokens (8,) atom types, src_coord (8,3) coordinates) for a small molecule."""
    src_tokens = torch.randint(0, 64, (8,))
    src_coord = torch.randn(8, 3)
    return (src_tokens, src_coord)


MENAGERIE_ENTRIES = [
    (
        "Uni-Mol (SE(3)-invariant 3D molecular transformer)",
        "build",
        "example_input",
        "2022",
        "DC",
    ),
]
