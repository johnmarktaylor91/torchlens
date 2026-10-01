"""MGNO: Multipole Graph Neural Operator.

Li et al., 2020.
Paper: https://arxiv.org/abs/2006.09535
Source: https://github.com/zongyi-li/multipole-graph-neural-operator

MGNO uses a MULTI-LEVEL GRAPH KERNEL (multipole method) for neural operator learning.
Inspired by the fast multipole method (FMM), it builds a hierarchy of graphs:
  - Fine level: dense local interactions (nearby points)
  - Coarse levels: long-range interactions via aggregated "multipole" nodes

Architecture:
  1. Input lifting: point cloud (x_i, f(x_i)) -> feature embedding
  2. Multi-level message passing:
     a. DOWNWARD (coarsening): aggregate fine-level features to coarser levels
        via learned pooling (e.g., K-means-like soft assignment)
     b. COARSE KERNEL: a graph convolution / message-passing on the coarse graph
        (long-range interactions)
     c. UPWARD (refinement): broadcast coarse features back to fine level
        + local fine-level message passing
  3. Output projection: node features -> function values

The key: the multi-level structure captures both LOCAL (fine) and GLOBAL (coarse)
interactions efficiently, like the multipole expansion in N-body problems.

Here we implement a simplified MGNO with 2 levels:
  - Level 0 (fine): N nodes, local k-nearest-neighbor message passing
  - Level 1 (coarse): M<<N aggregated nodes, full graph message passing
Downward/upward via learned soft assignment matrix (M x N).

Simplifications: N=32 input nodes, M=8 coarse nodes, 2 MGNO blocks,
d_model=32 features. No geometric preprocessing; random adjacency simulated by
fixed k-NN on random coords. Features are node-local.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    """Simple edge-feature message passing: aggregate over k-NN neighbors."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 4) -> None:
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        # edge_index: (N, k) -- for each node, k neighbor indices
        B, N, d = x.shape
        k = self.k

        # Gather neighbor features
        # edge_index: (N, k) -> (B, N, k, d)
        idx = edge_index.unsqueeze(0).unsqueeze(-1).expand(B, N, k, d)
        nbr = torch.gather(x.unsqueeze(2).expand(B, N, N, d), 2, idx)  # (B, N, k, d)

        # Edge features: [x_i, x_j - x_i]
        x_rep = x.unsqueeze(2).expand(B, N, k, d)
        edge_feat = torch.cat([x_rep, nbr - x_rep], dim=-1)  # (B, N, k, 2d)

        # MLP + max-pool over k neighbors
        msg = self.mlp(edge_feat)  # (B, N, k, out_dim)
        agg = msg.max(dim=2).values  # (B, N, out_dim)
        return agg


class CoarseKernelConv(nn.Module):
    """Full-graph kernel convolution on coarse nodes (all-to-all attention)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, M, d_model) -- coarse level nodes
        # Full kernel: sum_j W(z_i, z_j) v_j
        # Simplify: attention-weighted sum (no coords needed at coarse level)
        s = self.W_s(z)  # (B, M, d)
        r = self.W_r(z)  # (B, M, d)
        A = torch.softmax(s @ r.transpose(-2, -1) / (z.size(-1) ** 0.5), dim=-1)  # (B, M, M)
        agg = A @ z  # (B, M, d)
        return self.out(agg)


class MGNOBlock(nn.Module):
    """One MGNO block: fine->coarse->coarse-conv->fine + fine local conv."""

    def __init__(self, d_model: int, n_fine: int, n_coarse: int, k: int = 4) -> None:
        super().__init__()
        self.n_fine = n_fine
        self.n_coarse = n_coarse

        # Soft assignment: N fine -> M coarse (learned linear projection over fine dim)
        self.assign = nn.Linear(n_fine, n_coarse, bias=False)

        # Coarse-level operator
        self.coarse_conv = CoarseKernelConv(d_model)

        # Fine-level local message passing
        self.fine_conv = EdgeConv(d_model, d_model, k=k)
        self.k = k

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        B, N, d = x.shape

        # -- Coarse path --
        # Soft assignment: (B, N, d) -> (B, M, d)
        # assign weight A: (M, N); x_t: (B, d, N) -> (B, d, M)
        x_t = x.transpose(1, 2)  # (B, d, N)
        A = torch.softmax(self.assign.weight, dim=-1)  # (M, N) -- normalize over fine
        coarse = (A @ x_t.transpose(1, 2)).transpose(1, 2)  # actually:
        # Let's do: coarse = A (M,N) x x(B,N,d) -> (B,M,d)
        coarse = torch.einsum("mn,bnd->bmd", A, x)  # (B, M, d)

        # Coarse kernel convolution
        coarse_out = self.coarse_conv(self.norm1(coarse))

        # Upward: broadcast coarse -> fine (transpose assignment)
        fine_from_coarse = torch.einsum("mn,bmd->bnd", A, coarse_out)  # (B, N, d)

        # -- Fine local path --
        fine_local = self.fine_conv(x, edge_index)

        # Combine: residual + fine + coarse contribution
        h = x + fine_local + fine_from_coarse
        h = h + self.ffn(self.norm2(h))
        return h


class MGNO(nn.Module):
    """Multipole Graph Neural Operator.

    Operates on an unstructured point cloud with multi-level message passing.
    """

    def __init__(
        self,
        in_channels: int = 3,  # (f(x), x, y) -- function + coords
        out_channels: int = 1,
        d_model: int = 32,
        n_fine: int = 32,  # fine-level nodes
        n_coarse: int = 8,  # coarse-level aggregated nodes
        n_layers: int = 2,
        k_fine: int = 4,  # k-NN neighbors at fine level
    ) -> None:
        super().__init__()
        self.n_fine = n_fine
        self.k_fine = k_fine

        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList(
            [MGNOBlock(d_model, n_fine, n_coarse, k=k_fine) for _ in range(n_layers)]
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    @staticmethod
    def _build_knn_graph(coords: torch.Tensor, k: int) -> torch.Tensor:
        """Build k-NN graph from 2D coordinates. Returns edge_index (N, k)."""
        N = coords.size(1)
        # Use first batch element for graph structure (shared across batch)
        c = coords[0]  # (N, 2)
        diff = c.unsqueeze(0) - c.unsqueeze(1)  # (N, N, 2)
        dist2 = (diff**2).sum(-1)  # (N, N)
        dist2.fill_diagonal_(1e9)
        _, idx = dist2.topk(k, dim=-1, largest=False)  # (N, k)
        return idx

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_channels - 2) -- function values
        # coords: (B, N, 2)
        inp = torch.cat([x, coords], dim=-1)  # (B, N, in_channels)
        h = self.input_proj(inp)

        edge_index = self._build_knn_graph(coords, self.k_fine)  # (N, k)

        for blk in self.blocks:
            h = blk(h, edge_index)

        return self.output_mlp(h)  # (B, N, out_channels)


def build_mgno() -> nn.Module:
    return MGNO(
        in_channels=3, out_channels=1, d_model=32, n_fine=32, n_coarse=8, n_layers=2, k_fine=4
    )


def example_input_mgno():
    # function values (B=1, N=32, 1) + coordinates (B=1, N=32, 2)
    f = torch.randn(1, 32, 1)
    c = torch.rand(1, 32, 2)
    return [f, c]


MENAGERIE_ENTRIES = [
    ("MGNO (Multipole Graph Neural Operator)", "build_mgno", "example_input_mgno", "2020", "DC"),
]
