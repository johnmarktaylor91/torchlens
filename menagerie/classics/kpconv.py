"""KPConv: Flexible and Deformable Convolution for Point Clouds.

Thomas et al., ICCV 2019.
Paper: https://arxiv.org/abs/1904.08889
Source: https://github.com/HuguesTHOMAS/KPConv-PyTorch

KPConv (Kernel Point Convolution) defines convolution directly on 3D point sets.
A convolution kernel consists of K *kernel points* placed at learned (or fixed
geometric) positions in 3D space.  For each input point p, each kernel point k
contributes a weighted linear transformation of the neighboring features, where
the weight is determined by the spatial correlation between the kernel point and
the neighbor relative positions (a linear decay function of distance):

    h(p) = sum_{q in N(p)} sum_{k} h_sigma(||q - p - tilde_p_k||) * W_k * f(q)

where h_sigma is max(0, 1 - d/sigma) (triangular kernel, radius sigma).
This is the *rigid* KPConv variant where kernel points are fixed.  The
*deformable* variant adds a small MLP that predicts per-point kernel-point offsets.

Two entry points:
  KPConvRigidNet  -- stacked rigid KPConv layers + global pooling + classifier
  KPConvDeformableNet -- stacked deformable KPConv layers (extra offset MLP per layer)

Faithful compact reimplementation with pure-torch kNN (no CUDA ops).  Small
point cloud (64 pts, 2 layers, K=15 kernel points, sigma=0.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------


def _knn_idx(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """(B, N, k) indices of k-nearest neighbors for each point."""
    dists = torch.cdist(xyz, xyz)  # (B, N, N)
    return dists.topk(k, dim=-1, largest=False)[1]  # (B, N, k)


def _gather(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather xyz by idx.  xyz: (B, N, 3); idx: (B, N, k) -> (B, N, k, 3)."""
    B, N, k = idx.shape
    batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1)
    return xyz[batch_idx, idx]  # (B, N, k, 3)


def _gather_feats(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feats: (B, N, C); idx: (B, N, k) -> (B, N, k, C)."""
    B, N, k = idx.shape
    batch_idx = torch.arange(B, device=feats.device).view(B, 1, 1)
    return feats[batch_idx, idx]  # (B, N, k, C)


def _init_kernel_points(K: int, D: float = 1.0) -> torch.Tensor:
    """Place K kernel points roughly on a unit sphere (deterministic)."""
    if K == 1:
        return torch.zeros(1, 3)
    pts = []
    # origin
    pts.append(torch.zeros(3))
    # evenly distributed on a sphere via golden angle
    for i in range(1, K):
        theta = math.acos(1 - 2 * i / (K - 1)) if K > 2 else math.pi / 2
        phi = math.pi * (1 + math.sqrt(5)) * i
        pts.append(
            torch.tensor(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
            * D
        )
    return torch.stack(pts)  # (K, 3)


# -----------------------------------------------------------------------
# Rigid KPConv layer
# -----------------------------------------------------------------------


class KPConvLayer(nn.Module):
    """Single rigid KPConv layer.

    Args:
        in_ch: input feature channels (0 = use xyz as features).
        out_ch: output feature channels.
        K: number of kernel points.
        sigma: correlation radius (triangular decay).
        k_nn: number of neighbors per point.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        K: int = 15,
        sigma: float = 0.1,
        k_nn: int = 16,
    ) -> None:
        super().__init__()
        self.K = K
        self.sigma = sigma
        self.k_nn = k_nn
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Kernel weights: K linear maps of (in_ch, out_ch)
        self.kernel_weights = nn.Parameter(torch.randn(K, in_ch, out_ch) * 0.02)
        # Fixed kernel point positions (not optimised in rigid variant)
        kpts = _init_kernel_points(K, D=sigma)
        self.register_buffer("kernel_points", kpts)  # (K, 3)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3); feats: (B, N, in_ch) -> (B, N, out_ch)."""
        B, N, _ = xyz.shape
        idx = _knn_idx(xyz, self.k_nn)  # (B, N, k)
        nbr_xyz = _gather(xyz, idx)  # (B, N, k, 3)
        nbr_feats = _gather_feats(feats, idx)  # (B, N, k, in_ch)

        # diff: distance from each neighbor to each kernel point (kernel pts centered at center point)
        # nbr_xyz: (B, N, k, 3); kernel_points at center: center + kpt_offset
        # diff[b,n,q,k,:] = nbr_xyz[b,n,q,:] - (xyz[b,n,:] + kernel_points[k,:])
        kpts = self.kernel_points.view(1, 1, 1, self.K, 3)  # (1,1,1,K,3)
        center_exp = xyz.unsqueeze(2).unsqueeze(3)  # (B,N,1,1,3)
        nbr_exp = nbr_xyz.unsqueeze(3)  # (B,N,k,1,3)
        diff = nbr_exp - (center_exp + kpts)  # (B,N,k,K,3)
        # Triangular correlation: h(d) = max(0, 1 - d/sigma)
        d = diff.norm(dim=-1)  # (B, N, k, K)
        corr = F.relu(1.0 - d / (self.sigma + 1e-8))  # (B, N, k, K)

        # Weighted feature aggregation:
        # sum_k sum_q corr[q,k] * W[k] * f[q]
        # corr: (B,N,k,K); nbr_feats: (B,N,k,in_ch); W: (K,in_ch,out_ch)
        # -> (B,N,out_ch)
        # Einsum: corr (B,N,k,K) * nbr_feats (B,N,k,C) -> weighted (B,N,k,K,C)
        weighted = corr.unsqueeze(-1) * nbr_feats.unsqueeze(3)  # (B,N,k,K,in_ch)
        # Sum over neighbors k: (B,N,K,in_ch)
        agg = weighted.sum(dim=2)  # (B,N,K,in_ch)
        # Apply kernel weights per K: agg (B,N,K,in_ch) x W (K,in_ch,out_ch)
        out = torch.einsum("bnki,kio->bno", agg, self.kernel_weights)  # (B,N,out_ch)
        # BN expects (B, C, N) or (B*N, C)
        out = self.bn(out.view(B * N, self.out_ch)).view(B, N, self.out_ch)
        return F.relu(out, inplace=True)


# -----------------------------------------------------------------------
# Deformable KPConv layer
# -----------------------------------------------------------------------


class KPConvDeformableLayer(nn.Module):
    """Deformable KPConv: kernel points are offset per-point via a small MLP."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        K: int = 15,
        sigma: float = 0.1,
        k_nn: int = 16,
    ) -> None:
        super().__init__()
        self.K = K
        self.sigma = sigma
        self.k_nn = k_nn
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_weights = nn.Parameter(torch.randn(K, in_ch, out_ch) * 0.02)
        kpts = _init_kernel_points(K, D=sigma)
        self.register_buffer("kernel_points", kpts)  # (K, 3)
        # Offset network: local aggregated feat -> K * 3 offsets
        self.offset_net = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch, K * 3),
        )
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        idx = _knn_idx(xyz, self.k_nn)
        nbr_xyz = _gather(xyz, idx)  # (B,N,k,3)
        nbr_feats = _gather_feats(feats, idx)  # (B,N,k,in_ch)
        # Predict per-point kernel offsets from averaged neighbor feats
        avg_feat = nbr_feats.mean(dim=2)  # (B,N,in_ch)
        offsets = self.offset_net(avg_feat)  # (B,N,K*3)
        offsets = offsets.view(B, N, self.K, 3) * self.sigma  # scale

        # Deformed kernel positions per point: base + offsets
        kpts = self.kernel_points.view(1, 1, self.K, 3) + offsets  # (B,N,K,3)

        # Correlation between each neighbor and each deformed kernel point
        # kpts: (B,N,K,3) centered at each point; we need (B,N,k,K,3)
        center_kpts = xyz.unsqueeze(2).unsqueeze(3) + kpts.unsqueeze(2)  # (B,N,1,K,3) -> broadcast
        nbr_exp = nbr_xyz.unsqueeze(3)  # (B,N,k,1,3)
        diff = nbr_exp - center_kpts  # (B,N,k,K,3)
        d = diff.norm(dim=-1)  # (B,N,k,K)
        corr = F.relu(1.0 - d / (self.sigma + 1e-8))

        weighted = corr.unsqueeze(-1) * nbr_feats.unsqueeze(3)  # (B,N,k,K,in_ch)
        agg = weighted.sum(dim=2)  # (B,N,K,in_ch)
        out = torch.einsum("bnki,kio->bno", agg, self.kernel_weights)
        out = self.bn(out.view(B * N, self.out_ch)).view(B, N, self.out_ch)
        return F.relu(out, inplace=True)


# -----------------------------------------------------------------------
# Full classification networks
# -----------------------------------------------------------------------


class KPConvRigidNet(nn.Module):
    """Stacked rigid KPConv layers + global max-pool + classifier."""

    def __init__(self, num_classes: int = 40, k_nn: int = 16) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, 32)  # project xyz coords to initial features
        self.layer1 = KPConvLayer(32, 64, K=15, sigma=0.2, k_nn=k_nn)
        self.layer2 = KPConvLayer(64, 128, K=15, sigma=0.4, k_nn=k_nn)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3)
        feats = F.relu(self.input_proj(xyz))
        feats = self.layer1(xyz, feats)
        feats = self.layer2(xyz, feats)
        global_feat = feats.max(dim=1)[0]
        return self.classifier(global_feat)


class KPConvDeformableNet(nn.Module):
    """Stacked deformable KPConv layers + global max-pool + classifier."""

    def __init__(self, num_classes: int = 40, k_nn: int = 16) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, 32)
        self.layer1 = KPConvDeformableLayer(32, 64, K=15, sigma=0.2, k_nn=k_nn)
        self.layer2 = KPConvDeformableLayer(64, 128, K=15, sigma=0.4, k_nn=k_nn)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        feats = F.relu(self.input_proj(xyz))
        feats = self.layer1(xyz, feats)
        feats = self.layer2(xyz, feats)
        global_feat = feats.max(dim=1)[0]
        return self.classifier(global_feat)


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_rigid() -> nn.Module:
    """Rigid KPConv classifier (fixed kernel points, triangular correlation)."""
    return KPConvRigidNet(num_classes=40, k_nn=12)


def build_deformable() -> nn.Module:
    """Deformable KPConv classifier (per-point learned kernel-point offsets)."""
    return KPConvDeformableNet(num_classes=40, k_nn=12)


def example_input() -> torch.Tensor:
    """Small point cloud (1, 64, 3)."""
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "KPConv rigid (kernel-point triangular correlation, fixed geometry)",
        "build_rigid",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "KPConv deformable (per-point learned kernel-point offsets)",
        "build_deformable",
        "example_input",
        "2019",
        "DC",
    ),
]
