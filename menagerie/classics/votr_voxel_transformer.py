"""VOTR: Voxel Transformer backbone for 3D object detection (sparse voxel self-attn).

Mao, Xue, Niu, Bai, Feng, Liang, Xu & Xu, ICCV 2021, arXiv:2109.02497.
Source: https://github.com/PointsCoder/VOTR  (the sparse-voxel-attention CUDA ops:
hash-table build + k-neighbour gather over non-empty voxels).

VOTR replaces the 3D-convolution backbone of voxel detectors with a Transformer that
attends between voxels.  Its two building blocks:
  - **Submanifold Voxel Module (SVM)**: multi-head self-attention computed STRICTLY on
    the non-empty (occupied) voxels -- each query voxel attends to a gathered set of
    neighbouring occupied voxels (keeps the sparsity pattern, "submanifold").
  - **Sparse Voxel Module (SpVM)**: like SVM but can also produce features at empty
    voxel locations (expands the active set), used for downsampling.
  Each uses two attention ranges: **local attention** (immediate voxel neighbourhood,
  fine detail) and **dilated attention** (a few voxels reached with a growing stride,
  large receptive field at low cost).  Positional encodings come from relative voxel
  coordinates.

The CEILING in the menagerie is VOTR's custom CUDA sparse-voxel ops: a GPU hash table
over occupied voxel coordinates plus a kernel that gathers each query's local /
dilated non-empty neighbours.  These are OPTIMIZATIONS of plain tensor ops -- the hash
table is a coordinate->index map, the neighbour gather is index arithmetic on voxel
coordinates plus an occupancy mask, and the attention itself is standard scaled
dot-product attention restricted to the gathered (and masked-empty) neighbours.  This
module reimplements the FAITHFUL architecture (a submanifold local+dilated voxel
self-attention block over a small fixed voxel tensor with an occupancy mask) in pure
torch, so it traces and renders.

Small fixed grid: a (D x H x W) = (4 x 8 x 8) voxel volume with a random occupancy
mask; local window + one dilated step; 2 attention heads.  Single voxel-transformer
block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _neighbour_offsets(window: int, dilation: int) -> list[tuple[int, int, int]]:
    """Local (radius=window) + dilated (one step at ``dilation``) 3D voxel offsets."""
    offs: list[tuple[int, int, int]] = []
    r = window
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offs.append((dz, dy, dx))
    # Dilated attention: a few far voxels reached with a growing stride.
    for dz in (-dilation, dilation):
        offs.append((dz, 0, 0))
    for dy in (-dilation, dilation):
        offs.append((0, dy, 0))
    for dx in (-dilation, dilation):
        offs.append((0, 0, dx))
    return offs


class SparseVoxelAttention(nn.Module):
    """Submanifold multi-head self-attention over local + dilated voxel neighbours.

    A query voxel attends to its gathered neighbour voxels; empty (unoccupied)
    neighbours are masked out of the softmax (the submanifold restriction).
    """

    def __init__(self, dim: int, n_heads: int = 2, window: int = 1, dilation: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.offsets = _neighbour_offsets(window, dilation)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        # Relative-position encoding: one learned bias per neighbour offset per head.
        self.pos_bias = nn.Parameter(torch.zeros(len(self.offsets), n_heads))

    def forward(self, feat: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, D, H, W) dense voxel features; occ: (B, 1, D, H, W) occupancy {0,1}.
        B, C, D, H, W = feat.shape
        K = len(self.offsets)

        # Gather neighbour features and their occupancy by rolling the volume.
        neigh = []
        neigh_occ = []
        for dz, dy, dx in self.offsets:
            rolled = torch.roll(feat, shifts=(dz, dy, dx), dims=(2, 3, 4))
            rolled_occ = torch.roll(occ, shifts=(dz, dy, dx), dims=(2, 3, 4))
            neigh.append(rolled)
            neigh_occ.append(rolled_occ)
        neigh = torch.stack(neigh, dim=1)  # (B, K, C, D, H, W)
        neigh_occ = torch.stack(neigh_occ, dim=1)  # (B, K, 1, D, H, W)

        # Flatten voxels to a token axis N = D*H*W.
        N = D * H * W
        q_in = feat.reshape(B, C, N).transpose(1, 2)  # (B, N, C)
        kv_in = neigh.reshape(B, K, C, N).permute(0, 3, 1, 2)  # (B, N, K, C)
        occ_mask = neigh_occ.reshape(B, K, 1, N).permute(0, 3, 1, 2)  # (B, N, K, 1)

        q = self.q(q_in).reshape(B, N, self.n_heads, self.hd)
        k = self.k(kv_in).reshape(B, N, K, self.n_heads, self.hd)
        v = self.v(kv_in).reshape(B, N, K, self.n_heads, self.hd)

        # Scaled dot-product attention over the K gathered neighbours, per head.
        attn = torch.einsum("bnhd,bnkhd->bnkh", q, k) / (self.hd**0.5)
        attn = attn + self.pos_bias.view(1, 1, K, self.n_heads)
        # Mask empty neighbours out of the softmax (submanifold restriction).
        empty = (occ_mask < 0.5).expand(B, N, K, self.n_heads)
        attn = attn.masked_fill(empty, float("-inf"))
        attn = torch.softmax(attn, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)  # voxels with no occupied neighbour

        out = torch.einsum("bnkh,bnkhd->bnhd", attn, v).reshape(B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)
        # Keep features only on occupied voxels (submanifold output).
        return out * occ


class VoxelTransformerBlock(nn.Module):
    """One VOTR submanifold block: sparse voxel self-attention + FFN, with residuals."""

    def __init__(self, dim: int = 32, n_heads: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = SparseVoxelAttention(dim, n_heads=n_heads)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim * 2, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C+1, D, H, W) -- last channel is the occupancy mask.
        feat, occ = x[:, :-1], x[:, -1:]
        feat = feat + self.attn(self.norm1(feat), occ)
        feat = feat + self.ffn(self.norm2(feat)) * occ
        return feat


def build_votr() -> nn.Module:
    """Build one VOTR submanifold voxel-transformer block (dim=32, local+dilated attn)."""
    return VoxelTransformerBlock(dim=32, n_heads=2)


def example_input() -> torch.Tensor:
    """Voxel volume ``(1, 33, 4, 8, 8)``: 32 feature channels + 1 occupancy mask channel."""
    gen = torch.Generator().manual_seed(0)  # deterministic occupancy for stable renders
    feat = torch.randn(1, 32, 4, 8, 8, generator=gen)
    # Sparse occupancy (~40% voxels occupied), as the last channel.
    occ = (torch.rand(1, 1, 4, 8, 8, generator=gen) < 0.4).float()
    return torch.cat([feat, occ], dim=1)


MENAGERIE_ENTRIES = [
    (
        "VOTR (Voxel Transformer, submanifold sparse-voxel self-attention)",
        "build_votr",
        "example_input",
        "2021",
        "DC",
    ),
]
