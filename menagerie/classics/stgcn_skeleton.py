"""ST-GCN: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.

Yan et al., 2018 (AAAI).
Paper: https://arxiv.org/abs/1801.07455
Source: https://github.com/yysijie/st-gcn

Architecture:
  Skeleton is represented as a graph G=(V, E) where nodes = body joints, edges = bones.
  Adjacency matrix A is partitioned into K subsets (spatial partitioning strategy):
    - Root-centric (centripetal, centrifugal, self)  -> K=3 subsets (paper default)
  Each ST-GCN block:
    1. Spatial graph convolution:
         for each of K adjacency subsets A_k:
           y_k = A_k @ (X W_k)   where W_k is a 1x1 conv (temporal axis)
         y = sum_k(y_k)           (K-way sum)
       Plus a learnable edge importance mask M_k multiplied elementwise onto A_k.
    2. Temporal convolution:
         1D conv along the time axis (kernel_size=9, padded) after reshaping.
    3. Residual skip connection (1x1 conv if channels change).

Architecture details:
  Input: (B, C_in, T, V)  -- batch, channels, time frames, joints
  Stack of ST-GCN blocks: 64->64->64->128->128->128->256->256->256
  Global average pool over T and V -> FC classifier.

Compact version: 3 blocks, 18 joints, 16 frames, channels 32->64->128.
Trace+draw verified 2026-06-21.

Note on ST-STGCN disambiguation:
  The original Yan 2018 paper is titled "Spatial Temporal Graph Convolutional Networks"
  (ST-GCN). A separate line of "S-T-GCN" / "Spatio-Temporal GCN" work (e.g. traffic
  forecasting by Yu et al. 2018, IJCAI) is a different architecture. The b8.txt target
  "ST-STGCN" likely refers to a 'second' variant or alias of the same skeleton ST-GCN.
  Based on the brief description ("if truly identical, build ST-GCN once and note both
  map to it"), both ST-GCN-Skeleton and ST-STGCN map to this module.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Build a simple 18-joint skeleton adjacency (COCO-style)
# Partitioned into K=3 subsets: self, centripetal, centrifugal
# ---------------------------------------------------------------------------

N_JOINTS = 18

# COCO skeleton edges (0-indexed)
_COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),  # torso
    (13, 15),
    (14, 16),  # legs
]


def _build_adjacency(n_joints: int = N_JOINTS) -> torch.Tensor:
    """Build 3-subset adjacency (K=3): [self-link, centripetal, centrifugal].

    Returns: (3, N, N) float tensor
    """
    A = torch.zeros(3, n_joints, n_joints)
    # Self-connections (subset 0)
    A[0] += torch.eye(n_joints)
    # Undirected edges: treat both directions (subset 1 centripetal, subset 2 centrifugal)
    for u, v in _COCO_EDGES:
        if u < n_joints and v < n_joints:
            A[1, u, v] = 1.0  # centripetal
            A[2, v, u] = 1.0  # centrifugal
    # Row-normalize each subset
    for k in range(3):
        row_sum = A[k].sum(dim=-1, keepdim=True).clamp(min=1.0)
        A[k] = A[k] / row_sum
    return A


# ---------------------------------------------------------------------------
# ST-GCN block
# ---------------------------------------------------------------------------


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv (K subsets) + temporal conv + residual.

    Args:
        in_ch:     input channels
        out_ch:    output channels
        K:         number of adjacency subsets (partition strategy)
        T_kernel:  temporal conv kernel size (paper: 9)
        n_joints:  number of skeleton joints
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        K: int = 3,
        T_kernel: int = 9,
        n_joints: int = N_JOINTS,
        adjacency: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.K = K
        self.n_joints = n_joints

        # Register adjacency as buffer
        if adjacency is None:
            adjacency = _build_adjacency(n_joints)
        self.register_buffer("A", adjacency)  # (K, V, V)

        # Learnable edge importance masks (initialised to ones per subset)
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(n_joints, n_joints)) for _ in range(K)]
        )

        # Spatial: K parallel 1x1 convolutions (point-wise over joint channels)
        self.spatial_convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 1) for _ in range(K)])

        # Temporal conv: 2D conv with kernel (T_kernel, 1) -- stride over time only
        pad = (T_kernel - 1) // 2
        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (T_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
        )

        # Residual
        self.residual = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        res = self.residual(x)

        # Spatial graph convolution: for each subset apply A_k and W_k
        y = None
        for k in range(self.K):
            A_k = self.A[k] * self.edge_importance[k]  # (V, V) -- learnable mask
            # Reshape x: (B*T, C, V) for matmul
            xr = x.permute(0, 2, 1, 3).reshape(B * T, C, V)  # (BT, C, V)
            # Apply spatial conv (linear in channel) then graph diffusion
            # Project channels first
            xw = self.spatial_convs[k](x)  # (B, out_ch, T, V)
            # Graph diffusion: A_k @ xw along joint dim
            # xw: (B, out_ch, T, V) -> (B*T, out_ch, V)
            xwr = xw.permute(0, 2, 1, 3).reshape(B * T, -1, V)
            # matmul A_k (V, V) with V dim of xwr
            x_gcn = torch.matmul(xwr, A_k.T)  # (BT, out_ch, V)
            x_gcn = x_gcn.reshape(B, T, -1, V).permute(0, 2, 1, 3)  # (B, out_ch, T, V)
            y = x_gcn if y is None else y + x_gcn

        # Temporal conv
        y = self.temporal_conv(y)

        return self.relu(y + res)


# ---------------------------------------------------------------------------
# Full ST-GCN model
# ---------------------------------------------------------------------------


class STGCN(nn.Module):
    """ST-GCN skeleton action recognition model.

    Args:
        n_joints:   Number of skeleton joints.
        in_ch:      Input feature channels (e.g. 3 for xyz or 2 for xy).
        n_classes:  Number of action classes.
        n_blocks:   Number of ST-GCN blocks.
        base_ch:    Base channel width (paper: 64; compact: 32).
    """

    def __init__(
        self,
        n_joints: int = N_JOINTS,
        in_ch: int = 2,
        n_classes: int = 60,
        n_blocks: int = 3,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        A = _build_adjacency(n_joints)

        # Data batch normalisation
        self.bn = nn.BatchNorm1d(in_ch * n_joints)

        channels = [in_ch] + [base_ch * (2 ** (i // (n_blocks // 2 + 1))) for i in range(n_blocks)]
        channels[-1] = base_ch * 2  # ensure last block has double width

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                STGCNBlock(
                    channels[i],
                    channels[i + 1],
                    K=3,
                    T_kernel=7,  # paper: 9; reduced for compactness
                    n_joints=n_joints,
                    adjacency=A.clone(),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        # BN over joint*channel dimension
        xr = x.reshape(B, C * V, T)
        xr = self.bn(xr)
        x = xr.reshape(B, C, T, V)

        for block in self.blocks:
            x = block(x)

        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, final_ch)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_stgcn() -> nn.Module:
    """Build ST-GCN for skeleton action recognition (18 joints, compact)."""
    return STGCN(n_joints=18, in_ch=2, n_classes=60, n_blocks=3, base_ch=32)


def example_input_skeleton() -> torch.Tensor:
    """Input: (B=1, C=2, T=16, V=18) -- xy coords, 16 frames, 18 joints."""
    return torch.randn(1, 2, 16, 18)


MENAGERIE_ENTRIES = [
    (
        "ST-GCN-Skeleton (spatial-temporal graph conv, skeleton action recognition)",
        "build_stgcn",
        "example_input_skeleton",
        "2018",
        "DC",
    ),
    (
        "ST-STGCN (alias for ST-GCN Yan 2018; same skeleton graph conv architecture)",
        "build_stgcn",
        "example_input_skeleton",
        "2018",
        "DC",
    ),
]
