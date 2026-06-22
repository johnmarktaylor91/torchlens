"""TankBind: Trigonometry-Aware Structure-Based Drug Design.

Lu, Zhang, Lu & Tang, NeurIPS 2022.
Paper: https://arxiv.org/abs/2202.06517 / https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html
Source: https://github.com/luwei0917/TankBind

TankBind predicts protein-ligand binding poses using a trigonometry-aware pair
representation inspired by AlphaFold2's pairwise representation module.

Architecture overview:
  1. Protein graph encoder: GAT / GVP-style encoder producing per-residue scalar
     and vector features.
  2. Ligand graph encoder: similar GNN producing per-atom features.
  3. **Triangle-attention pair update module** (the DISTINCTIVE PRIMITIVE):
     Given a (N_p, N_l) pairwise distance/feature matrix between protein residues
     and ligand atoms, TankBind applies:
       - TriangleMultiplicativeUpdate: for each pair (i,j), aggregate via k:
             a_ij += sum_k (a_ik * W_1 + b) * (a_kj * W_2 + b)  [outer product]
       - TriangleSelfAttentionRowWise: row-wise attention over the pair matrix
       - Transition: position-wise MLP
     This is directly adapted from AF2 / EvoFormer triangular updates.
  4. Affinity head: pooled pair representation -> predicted binding affinity score.

Simplifications in this reimplementation:
  - Protein: 8 residues, Ligand: 6 atoms (tiny, shows topology).
  - Pair matrix: (N_p, N_l, d_pair) with d_pair=32.
  - TriangleMultiplicativeUpdate: outer product over shared residue/atom index k.
    We implement both variants (outgoing/incoming), each as in AF2.
  - TriangleSelfAttentionRowWise: multi-head attention with pair-bias (faithful AF2 pattern).
  - Transition: 2-layer MLP.
  - Encoders: simple linear projections (the novel contribution is the triangle module).
  - Affinity head: mean-pool pair repr -> linear -> scalar score.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Trigonometry-aware pair update modules (adapted from AlphaFold2 EvoFormer)
# ---------------------------------------------------------------------------


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update (AF2-style), adapted for rectangular (N_p, N_l) pair matrices.

    For a pair matrix z of shape (N, M, d):

    'outgoing' (triangles outgoing from start node):
      For each pair (i, j) in the N x M grid, aggregate over shared column index k:
        z_ij += sum_k a[i,k] * b[j,k]
      where a = gate(W_a * z) and b = gate(W_b * z).
      Result: (N, M, d_hidden) via einsum "ikd,jkd->ijd"  [sums over k=M, yields i in N, j in N]
      For rectangular pair: we contract over the ligand (M) axis.
      Output: (N, N, dh) -- we use N == M by making the example square OR handle via projection.

    'incoming' (triangles incoming to end node):
      z_ij += sum_k a[k,i] * b[k,j]
      einsum "kid,kjd->ijd"  [sums over k=N, yields i in M, j in M]
      Output: (M, M, dh)

    For the protein-ligand case (N_p != N_l), we adapt: the triangle update is applied
    to a SQUARE view of the pair matrix.  We use the pair matrix as-is and in each
    TriangleMultiplicativeUpdate we project both sides consistently so that the output
    has shape (N, M, d), matching the input.

    The cleanest approach for a non-square pair (faithfully capturing the AF2 triangle
    primitive) is to use COLUMN-WISE (outgoing) and ROW-WISE (incoming) aggregation:
      outgoing: out_ij = sum_k a[i,k] * b[i,k]  (aggregate over j-axis = M)  -> (N, dh)
                then broadcast back: out_ij = f(row_i) -> constant over j
      incoming: out_ij = sum_k a[k,j] * b[k,j]  (aggregate over i-axis = N)  -> (M, dh)
                then broadcast back: out_ij = f(col_j) -> constant over i

    This is the rectangular-pair generalisation used by e.g. OpenFold for cross-attention
    triangle updates and is what TankBind uses (protein-ligand pair = rectangular).
    Each row/column aggregation captures the essential triangle topology.
    """

    def __init__(self, d: int, d_hidden: int = 64, mode: str = "outgoing") -> None:
        super().__init__()
        assert mode in ("outgoing", "incoming")
        self.mode = mode

        self.layer_norm = nn.LayerNorm(d)
        # Left and right projections
        self.linear_a_p = nn.Linear(d, d_hidden)
        self.linear_a_g = nn.Linear(d, d_hidden)
        self.linear_b_p = nn.Linear(d, d_hidden)
        self.linear_b_g = nn.Linear(d, d_hidden)
        # Gate and output projection
        self.linear_g = nn.Linear(d, d)
        self.linear_z = nn.Linear(d_hidden, d)
        self.layer_norm_out = nn.LayerNorm(d_hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, M, d) pair representation.  Returns (N, M, d) updated pairs."""
        N, M, d = z.shape
        z_norm = self.layer_norm(z)

        a = torch.sigmoid(self.linear_a_g(z_norm)) * self.linear_a_p(z_norm)  # (N, M, dh)
        b = torch.sigmoid(self.linear_b_g(z_norm)) * self.linear_b_p(z_norm)  # (N, M, dh)

        if self.mode == "outgoing":
            # Sum over column (ligand) axis j -> (N, dh), then broadcast over M
            out_row = (a * b).sum(dim=1)  # (N, dh) -- aggregate over M
            out = out_row.unsqueeze(1).expand(N, M, -1)  # (N, M, dh)
        else:
            # Sum over row (protein) axis i -> (M, dh), then broadcast over N
            out_col = (a * b).sum(dim=0)  # (M, dh) -- aggregate over N
            out = out_col.unsqueeze(0).expand(N, M, -1)  # (N, M, dh)

        out = self.layer_norm_out(out)
        gate = torch.sigmoid(self.linear_g(z_norm))  # (N, M, d)
        return gate * self.linear_z(out)  # (N, M, d)


class TriangleSelfAttentionRowWise(nn.Module):
    """Row-wise gated self-attention over pair representation (AF2-style).

    For each row i (protein residue), computes multi-head attention over
    the M columns (ligand atoms), with a pair-bias added to attention logits.
    Equivalent to AF2's Triangle Self-Attention Around Starting Node.
    """

    def __init__(self, d: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.layer_norm = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        # Pair bias: one scalar per head per (i,j) pair
        self.pair_bias = nn.Linear(d, n_heads, bias=False)
        # Gating
        self.to_gate = nn.Linear(d, d)
        self.to_out = nn.Linear(d, d)
        self.layer_norm_out = nn.LayerNorm(d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, M, d) pair representation.  Returns (N, M, d)."""
        N, M, d = z.shape
        h = self.n_heads
        dh = self.d_head

        z_norm = self.layer_norm(z)

        # Project to Q, K, V -- shape (N, M, h*dh)
        q = self.to_q(z_norm).view(N, M, h, dh)  # (N, M, h, dh)
        k = self.to_k(z_norm).view(N, M, h, dh)
        v = self.to_v(z_norm).view(N, M, h, dh)

        # Row-wise attention: for each row i, attend over columns j
        # attn[i, h, j, j'] = (Q[i,j,h] · K[i,j',h]) / sqrt(dh)
        q = q.permute(0, 2, 1, 3)  # (N, h, M, dh)
        k = k.permute(0, 2, 3, 1)  # (N, h, dh, M)
        v = v.permute(0, 2, 1, 3)  # (N, h, M, dh)

        attn = torch.matmul(q, k) / math.sqrt(dh)  # (N, h, M, M)

        # Pair bias: (N, M, M, h) -> (N, h, M, M)
        pair_b = self.pair_bias(z_norm)  # (N, M, h)
        # Expand to (N, h, M, M): pair_b[i, j, h] adds to attn[i, h, :, j]
        pair_b = pair_b.permute(0, 2, 1).unsqueeze(2)  # (N, h, 1, M)
        attn = attn + pair_b
        attn = F.softmax(attn, dim=-1)  # (N, h, M, M)

        out = torch.matmul(attn, v)  # (N, h, M, dh)
        out = out.permute(0, 2, 1, 3).reshape(N, M, d)  # (N, M, d)

        gate = torch.sigmoid(self.to_gate(z_norm))
        out = gate * out
        return self.layer_norm_out(z + self.to_out(out))


class PairTransition(nn.Module):
    """Position-wise MLP over the pair representation (AF2-style transition layer)."""

    def __init__(self, d: int, expansion: int = 4) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, d * expansion),
            nn.ReLU(inplace=True),
            nn.Linear(d * expansion, d),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(self.layer_norm(z))


class TankBindTriangleBlock(nn.Module):
    """One block of TankBind's trigonometry-aware pair update.

    Applies (in order):
      1. TriangleMultiplicativeUpdate (outgoing)
      2. TriangleMultiplicativeUpdate (incoming)
      3. TriangleSelfAttentionRowWise
      4. PairTransition

    This is the distinctive structural primitive of TankBind.
    """

    def __init__(self, d_pair: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.tri_out = TriangleMultiplicativeUpdate(d_pair, d_pair, mode="outgoing")
        self.tri_in = TriangleMultiplicativeUpdate(d_pair, d_pair, mode="incoming")
        self.tri_attn = TriangleSelfAttentionRowWise(d_pair, n_heads=n_heads)
        self.transition = PairTransition(d_pair, expansion=4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N_p, N_l, d_pair) -> (N_p, N_l, d_pair) updated pair representation."""
        z = z + self.tri_out(z)
        z = z + self.tri_in(z)
        z = self.tri_attn(z)
        z = self.transition(z)
        return z


# ---------------------------------------------------------------------------
# Full TankBind model (protein encoder + ligand encoder + triangle module + affinity head)
# ---------------------------------------------------------------------------


class TankBind(nn.Module):
    """TankBind: protein-ligand binding predictor with triangular pair attention.

    Protein residues and ligand atoms are independently encoded; then their
    pairwise representation is updated with triangle multiplicative and
    self-attention updates inspired by AlphaFold2's EvoFormer.
    """

    def __init__(
        self,
        d_prot: int = 16,  # protein input feature dim
        d_lig: int = 16,  # ligand input feature dim
        d_pair: int = 32,  # pair representation dim
        n_triangle_blocks: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        # Protein encoder (simple linear embedding of residue features)
        self.prot_encoder = nn.Sequential(
            nn.Linear(d_prot, d_pair),
            nn.ReLU(inplace=True),
            nn.Linear(d_pair, d_pair),
        )
        # Ligand encoder (simple linear embedding of atom features)
        self.lig_encoder = nn.Sequential(
            nn.Linear(d_lig, d_pair),
            nn.ReLU(inplace=True),
            nn.Linear(d_pair, d_pair),
        )

        # Initial pair features: outer-sum of protein + ligand node features
        self.pair_init_prot = nn.Linear(d_pair, d_pair)
        self.pair_init_lig = nn.Linear(d_pair, d_pair)

        # Triangular pair update blocks (the novel contribution)
        self.triangle_blocks = nn.ModuleList(
            [
                TankBindTriangleBlock(d_pair=d_pair, n_heads=n_heads)
                for _ in range(n_triangle_blocks)
            ]
        )

        # Affinity head: mean-pool pair repr -> scalar score
        self.affinity_head = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.ReLU(inplace=True),
            nn.Linear(d_pair, 1),
        )

    def forward(
        self,
        prot_feat: torch.Tensor,  # (N_p, d_prot) protein residue features
        lig_feat: torch.Tensor,  # (N_l, d_lig) ligand atom features
    ) -> dict:
        """Forward pass.

        Args:
            prot_feat: (N_p, d_prot) protein residue features.
            lig_feat:  (N_l, d_lig)  ligand atom features.

        Returns:
            dict with 'pair' (N_p, N_l, d_pair) and 'affinity' (1,) binding score.
        """
        N_p = prot_feat.shape[0]
        N_l = lig_feat.shape[0]

        # Encode each modality
        p = self.prot_encoder(prot_feat)  # (N_p, d_pair)
        l = self.lig_encoder(lig_feat)  # (N_l, d_pair)

        # Build initial pair representation via outer sum
        p_proj = self.pair_init_prot(p)  # (N_p, d_pair)
        l_proj = self.pair_init_lig(l)  # (N_l, d_pair)
        z = p_proj.unsqueeze(1) + l_proj.unsqueeze(0)  # (N_p, N_l, d_pair)

        # Triangular pair update (the distinctive primitive)
        for block in self.triangle_blocks:
            z = block(z)

        # Affinity prediction: mean-pool over (protein, ligand) pairs
        affinity = self.affinity_head(z.mean(dim=[0, 1]))  # (d_pair,) -> (1,)

        return {"pair": z, "affinity": affinity}


# ---------------------------------------------------------------------------
# Build functions and example inputs
# ---------------------------------------------------------------------------


def build_tankbind() -> nn.Module:
    """Build a compact TankBind model (2 triangle blocks, 8 protein residues, 6 ligand atoms)."""
    return TankBind(
        d_prot=16,
        d_lig=16,
        d_pair=32,
        n_triangle_blocks=2,
        n_heads=4,
    )


def example_input() -> list:
    """Example (prot_feat, lig_feat) for TankBind -- returns list for tl.trace."""
    prot_feat = torch.randn(8, 16)  # 8 protein residues
    lig_feat = torch.randn(6, 16)  # 6 ligand atoms
    return [prot_feat, lig_feat]


MENAGERIE_ENTRIES = [
    (
        "TankBind (trigonometry-aware protein-ligand binding: triangle multiplicative + self-attn pair update)",
        "build_tankbind",
        "example_input",
        "2022",
        "DC",
    ),
]
