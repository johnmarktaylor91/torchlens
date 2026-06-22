"""PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds.

Xu, Liu, et al., CVPR 2021.
Paper: https://arxiv.org/abs/2103.14635
Source: https://github.com/CVMI-Lab/PAConv

PAConv's distinctive primitive is *position-adaptive weight assembly*: instead of
a fixed set of kernel weights, PAConv learns a *weight bank* of M weight matrices,
and for each pair of points (center + neighbor) a small MLP on the relative
3D position predicts a softmax score vector over the M basis matrices.  The actual
convolution weight used for that pair is the score-weighted sum of the bank:

    W(delta_p) = sum_{m=1}^{M} s_m(delta_p) * W_m

where s_m are the softmax-normalized scores from the ScoreNet MLP.
The assembled W is then applied to the neighbor feature to produce the output
feature for that neighbor, and neighbor outputs are aggregated (max-pool).

This is a faithful compact random-init reimpl of PAConv as used in DGCNN-PAConv
(the paper's main model): replace EdgeConv with PAConv in a dynamic-graph backbone.
  - Weight bank: M=4 basis matrices (in_ch, out_ch) each.
  - ScoreNet: MLP(3 -> 8 -> M), softmax scores.
  - Dynamic kNN graph rebuilt per layer.
  - 2 PAConv layers + global max-pool + classifier.
  - Input: (B, N, 3) point cloud.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _knn_idx(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Dynamic kNN graph in feature space.  feats: (B, N, C) -> (B, N, k) indices."""
    dists = torch.cdist(feats, feats)
    return dists.topk(k, dim=-1, largest=False)[1]


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: (B, N, C); idx: (B, N, k) -> (B, N, k, C)."""
    B, N, k = idx.shape
    bi = torch.arange(B, device=x.device).view(B, 1, 1)
    return x[bi, idx]  # (B, N, k, C)


class ScoreNet(nn.Module):
    """MLP: relative position (3) -> M softmax scores (one per basis weight)."""

    def __init__(self, M: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, M),
        )
        self.M = M

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        # rel_pos: (..., 3) -> (..., M)
        return torch.softmax(self.net(rel_pos), dim=-1)


class PAConvLayer(nn.Module):
    """Single PAConv layer with M-basis adaptive weight assembly.

    Args:
        in_ch: input feature channels.
        out_ch: output feature channels.
        M: number of basis weight matrices in the weight bank.
        k: number of dynamic neighbors.
    """

    def __init__(self, in_ch: int, out_ch: int, M: int = 4, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.M = M
        # Weight bank: M basis matrices (in_ch -> out_ch)
        self.weight_bank = nn.Parameter(torch.randn(M, in_ch, out_ch) * 0.02)
        self.score_net = ScoreNet(M)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3); feats: (B, N, in_ch) -> (B, N, out_ch)."""
        B, N, _ = xyz.shape
        # Dynamic graph in current feature space
        idx = _knn_idx(feats, self.k)  # (B, N, k)
        nbr_xyz = _gather(xyz, idx)  # (B, N, k, 3)
        nbr_feats = _gather(feats, idx)  # (B, N, k, in_ch)
        rel_pos = nbr_xyz - xyz.unsqueeze(2)  # (B, N, k, 3)

        # Scores from ScoreNet: (B, N, k, M)
        scores = self.score_net(rel_pos)

        # Assemble per-pair weight: sum_m score[m] * W_m
        # scores: (B,N,k,M); weight_bank: (M, in_ch, out_ch)
        # assembled_W: (B, N, k, in_ch, out_ch)
        assembled_W = torch.einsum("bnkm,mio->bnkio", scores, self.weight_bank)

        # Apply assembled_W to neighbor features
        # nbr_feats: (B,N,k,in_ch) -> out: (B,N,k,out_ch)
        out = torch.einsum("bnki,bnkio->bnko", nbr_feats, assembled_W)

        # Max-pool over neighbors
        out = out.max(dim=2)[0]  # (B, N, out_ch)
        out = self.bn(out.view(B * N, self.out_ch)).view(B, N, self.out_ch)
        return F.relu(out, inplace=True)


class PAConvNet(nn.Module):
    """PAConv classification network (DGCNN-PAConv backbone)."""

    def __init__(self, num_classes: int = 40, k: int = 16, M: int = 4) -> None:
        super().__init__()
        self.layer1 = PAConvLayer(3, 64, M=M, k=k)
        self.layer2 = PAConvLayer(64, 128, M=M, k=k)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3) -- initial feats = xyz coords
        feats = self.layer1(xyz, xyz)
        feats = self.layer2(xyz, feats)
        global_feat = feats.max(dim=1)[0]
        return self.classifier(global_feat)


def build() -> nn.Module:
    """Build PAConv classifier (M=4 weight bank, k=16 dynamic neighbors)."""
    return PAConvNet(num_classes=40, k=16, M=4)


def example_input() -> torch.Tensor:
    """Small point cloud (1, 64, 3)."""
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PAConv (position-adaptive dynamic weight-bank assembly)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
