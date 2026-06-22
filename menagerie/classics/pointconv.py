"""PointConv: Deep Convolutional Networks on 3D Point Clouds.

Wu, Qi & Fuxin, CVPR 2019.
Paper: https://arxiv.org/abs/1811.07246
Source: https://github.com/DylanWuSee/pointconv_pytorch

PointConv approximates a continuous convolution over an unordered point set. For
each (sampled) center point it gathers a local kNN neighborhood, computes
relative coordinates, and applies:

* a *weight function* -- an MLP mapping the 3D relative offset of each neighbor
  to a per-feature continuous convolution weight;
* a *density reweighting* -- an MLP on the inverse local point density (here a
  kernel-density estimate) used to scale neighbor contributions;
* a final 1x1 linear that mixes the weighted, aggregated neighbor features.

This is a faithful random-init reimplementation of a PointConv set-abstraction
layer stack with pure-torch kNN (no CUDA ops). It classifies a small point cloud.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance between two point sets (B, N, 3)/(B, M, 3)."""
    return torch.cdist(src, dst) ** 2


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` nearest neighbors of each point (B, N, k)."""
    dist = square_distance(xyz, xyz)
    return dist.topk(k, dim=-1, largest=False)[1]


def kde_density(xyz: torch.Tensor, bandwidth: float = 0.1) -> torch.Tensor:
    """Gaussian kernel-density estimate at each point (B, N, 1)."""
    dist = square_distance(xyz, xyz)
    gauss = torch.exp(-dist / (2.0 * bandwidth * bandwidth))
    density = gauss.sum(dim=-1, keepdim=True)
    return density


class PointConvLayer(nn.Module):
    """A single PointConv layer with weight-function MLP + density reweighting."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 16, mid: int = 16) -> None:
        super().__init__()
        self.k = k
        # Weight function: relative xyz (3) -> per-element conv weights (mid).
        self.weight_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, mid),
        )
        # Density reweighting MLP on inverse density.
        self.density_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )
        # Final linear mixing aggregated weighted features.
        self.linear = nn.Linear((in_ch + 3) * mid, out_ch)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3), feats: (B, N, C)
        b, n, _ = xyz.shape
        idx = knn_indices(xyz, self.k)  # (B, N, k)
        batch_idx = torch.arange(b, device=xyz.device).view(b, 1, 1)
        grouped_xyz = xyz[batch_idx, idx]  # (B, N, k, 3)
        rel_xyz = grouped_xyz - xyz.unsqueeze(2)  # (B, N, k, 3)
        grouped_feats = feats[batch_idx, idx]  # (B, N, k, C)
        grouped = torch.cat([rel_xyz, grouped_feats], dim=-1)  # (B, N, k, C+3)

        conv_w = self.weight_mlp(rel_xyz)  # (B, N, k, mid)

        density = kde_density(xyz)  # (B, N, 1)
        inv_density = 1.0 / (density + 1e-6)
        dens_w = self.density_mlp(inv_density)  # (B, N, 1)
        grouped = grouped * dens_w.unsqueeze(2)

        # Outer product per neighbor: (B,N,k,C+3,1) x (B,N,k,1,mid) -> (B,N,k,C+3,mid).
        out = grouped.unsqueeze(-1) * conv_w.unsqueeze(-2)
        out = out.sum(dim=2)  # aggregate over neighbors -> (B, N, C+3, mid)
        out = out.reshape(b, n, -1)
        return F.relu(self.linear(out))


class PointConvClassifier(nn.Module):
    """Stacked PointConv layers + global pooling + classifier head."""

    def __init__(self, num_classes: int = 40, k: int = 16) -> None:
        super().__init__()
        self.layer1 = PointConvLayer(in_ch=3, out_ch=32, k=k)
        self.layer2 = PointConvLayer(in_ch=32, out_ch=64, k=k)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3). Initial features = coordinates.
        feats = self.layer1(xyz, xyz)
        feats = self.layer2(xyz, feats)
        pooled = feats.max(dim=1)[0]
        return self.classifier(pooled)


def build() -> nn.Module:
    """Build the PointConv point-cloud classifier."""
    return PointConvClassifier(num_classes=40, k=16)


def example_input() -> torch.Tensor:
    """Small point cloud ``(1, 64, 3)`` (few points keeps the kNN trace compact)."""
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PointConv (continuous-weight density-reweighted point convolution)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
