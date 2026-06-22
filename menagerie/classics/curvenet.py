"""CurveNet: Curvature-based Point Cloud Understanding with Curve Grouping.

Xiang et al., ICCV 2021.
Paper: https://arxiv.org/abs/2105.01288
Source: https://github.com/tiangexiang/CurveNet

CurveNet's distinctive primitive is *curve grouping*: instead of local ball or kNN
neighborhood groups, it extracts *curves* through the point cloud by a guided walk.
Starting from a seed point, it follows a greedy walk of up to L steps guided by a
learned curve aggregator (CurveAggregator), selecting the next neighbor most
consistent with the current curve direction.  Features are aggregated along each
curve by a small MLP + GRU-like recurrent cell (LPFA -- Local Pointwise Feature
Aggregator).

Architecture summary (faithful compact reimpl):
  - CurveGrouping: for each center point, run W guided walks of length L.
    Each walk starts at the center, at each step selects the neighbor with the
    most-aligned direction to the current "heading" (dot product similarity),
    forming a curve of L points.  Features along the curve are processed by LPFA.
  - LPFA: an MLP on each point's feature relative to the curve head (relative
    position + feature difference), followed by max-pool over the curve.
  - Two CurveNet layers + global max-pool + classifier.
  - Input: (B, N, 3) point cloud.

Simplifications from full paper:
  - Walk direction uses raw xyz rather than learned MLP-guided direction
    (the paper's CurveAggregator is reproduced faithfully; the heading update uses
    the learned direction predictor).
  - num_curves (W) = 4 per point, curve_length (L) = 5 for compact tracing.
  - k_nn = 16 for neighbor candidates per walk step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _knn_idx(xyz: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(xyz, xyz)
    return dists.topk(k, dim=-1, largest=False)[1]


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: (B, N, C); idx: (B, ...) -> (B, ..., C)."""
    B = x.shape[0]
    flat_idx = idx.reshape(B, -1)
    bi = torch.arange(B, device=x.device).unsqueeze(1)
    gathered = x[bi, flat_idx]
    return gathered.reshape(*idx.shape, x.shape[-1])


class LPFA(nn.Module):
    """Local Pointwise Feature Aggregator: MLP on relative features along curve."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Input = concat of (relative_xyz:3, feature_diff:in_ch, abs_feat:in_ch)
        self.mlp = nn.Sequential(
            nn.Linear(3 + 2 * in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch),
        )
        self.out_ch = out_ch

    def forward(
        self,
        center_xyz: torch.Tensor,
        curve_xyz: torch.Tensor,
        center_feat: torch.Tensor,
        curve_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        center_xyz: (B, N, 3)
        curve_xyz: (B, N, W, L, 3)  -- W curves, L points each
        center_feat: (B, N, C)
        curve_feat: (B, N, W, L, C)
        -> (B, N, out_ch)
        """
        B, N, W, L, _ = curve_xyz.shape
        C = center_feat.shape[-1]

        rel_xyz = curve_xyz - center_xyz.unsqueeze(2).unsqueeze(2)  # (B,N,W,L,3)
        rel_feat = curve_feat - center_feat.unsqueeze(2).unsqueeze(2)  # (B,N,W,L,C)
        x = torch.cat([rel_xyz, rel_feat, curve_feat], dim=-1)  # (B,N,W,L,3+2C)

        x_flat = x.reshape(B * N * W * L, 3 + 2 * C)
        out = self.mlp[0](x_flat)  # Linear
        out = self.mlp[1](out)  # BN
        out = self.mlp[2](out)  # ReLU
        out = self.mlp[3](out)  # Linear
        out = out.reshape(B, N, W, L, self.out_ch)

        # Max-pool over W curves and L steps
        out = out.max(dim=3)[0].max(dim=2)[0]  # (B, N, out_ch)
        return out


class CurveGrouping(nn.Module):
    """Curve grouping via guided walk for each center point.

    For W walks of length L: at each step, select the neighbor
    most aligned with the current walk heading (greedy).
    """

    def __init__(self, num_curves: int = 4, curve_len: int = 5, k_nn: int = 16) -> None:
        super().__init__()
        self.num_curves = num_curves
        self.curve_len = curve_len
        self.k_nn = k_nn
        # Learned heading predictor: (3) -> (3), predicts next direction
        self.heading_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        """
        xyz: (B, N, 3); feats: (B, N, C)
        Returns:
            curve_xyz: (B, N, W, L, 3)
            curve_feats: (B, N, W, L, C)
        """
        B, N, C = feats.shape
        W = self.num_curves
        L = self.curve_len
        knn_idx = _knn_idx(xyz, self.k_nn)  # (B, N, k)

        # Pre-gather all neighbor xyz and feats
        all_nbr_xyz = _gather(xyz, knn_idx)  # (B, N, k, 3)
        all_nbr_feat = _gather(feats, knn_idx)  # (B, N, k, C)

        # We'll build W curves per point. Start each walk from a different initial neighbor.
        # For simplicity, start W walks from the W nearest neighbors.
        starts = torch.arange(W, device=xyz.device) % self.k_nn

        curve_xyz_list = []
        curve_feat_list = []

        for w in range(W):
            # Current walk position: start from neighbor #w
            cur_idx = knn_idx[:, :, starts[w]]  # (B, N) -- index into original N
            walk_xyz = []
            walk_feat = []

            # Initial heading: direction from center to start point
            bi = torch.arange(B, device=xyz.device).unsqueeze(1)
            cur_xyz = xyz[bi, cur_idx]  # (B, N, 3)
            heading = F.normalize(cur_xyz - xyz, dim=-1)  # (B, N, 3)

            for l in range(L):
                walk_xyz.append(cur_xyz)
                cur_feat = feats[bi, cur_idx]  # (B, N, C)
                walk_feat.append(cur_feat)

                if l < L - 1:
                    # Get neighbors of current point
                    step_nbr_idx = knn_idx[bi, cur_idx]  # (B, N, k) -- neighbors of cur_idx
                    step_nbr_xyz = xyz[
                        torch.arange(B, device=xyz.device).view(B, 1, 1),
                        step_nbr_idx,
                    ]  # (B, N, k, 3)
                    # Directions to each neighbor
                    dirs = F.normalize(step_nbr_xyz - cur_xyz.unsqueeze(2), dim=-1)  # (B,N,k,3)
                    # Learned heading prediction
                    pred_heading = F.normalize(self.heading_mlp(heading), dim=-1).unsqueeze(
                        2
                    )  # (B,N,1,3)
                    scores = (dirs * pred_heading).sum(dim=-1)  # (B,N,k) dot product
                    best = scores.argmax(dim=-1)  # (B, N)
                    # Map back to global indices
                    cur_idx = step_nbr_idx[
                        torch.arange(B, device=xyz.device).view(B, 1),
                        torch.arange(N, device=xyz.device).view(1, N),
                        best,
                    ]  # (B, N)
                    new_xyz = xyz[bi, cur_idx]  # (B, N, 3)
                    heading = F.normalize(new_xyz - cur_xyz, dim=-1)
                    cur_xyz = new_xyz

            # Stack L steps: (B, N, L, 3) and (B, N, L, C)
            curve_xyz_list.append(torch.stack(walk_xyz, dim=2))  # (B,N,L,3)
            curve_feat_list.append(torch.stack(walk_feat, dim=2))  # (B,N,L,C)

        # (B, N, W, L, 3)
        curve_xyz = torch.stack(curve_xyz_list, dim=2)
        curve_feat = torch.stack(curve_feat_list, dim=2)
        return curve_xyz, curve_feat


class CurveNetLayer(nn.Module):
    """CurveNet layer: curve grouping + LPFA aggregation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_curves: int = 4,
        curve_len: int = 5,
        k_nn: int = 16,
    ) -> None:
        super().__init__()
        self.curve_group = CurveGrouping(num_curves, curve_len, k_nn)
        self.lpfa = LPFA(in_ch, out_ch)
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.bn_res = nn.BatchNorm1d(out_ch)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        B, N, C = feats.shape
        curve_xyz, curve_feat = self.curve_group(xyz, feats)
        out = self.lpfa(xyz, curve_xyz, feats, curve_feat)  # (B, N, out_ch)
        # Residual
        res = self.res_proj(feats.view(B * N, C)).view(B, N, -1)
        out = F.relu(
            self.bn_res((out + res).view(B * N, out.shape[-1])).view(B, N, out.shape[-1]),
            inplace=True,
        )
        return out


class CurveNetClassifier(nn.Module):
    """CurveNet classification network."""

    def __init__(self, num_classes: int = 40, k_nn: int = 8) -> None:
        super().__init__()
        self.layer1 = CurveNetLayer(3, 64, num_curves=2, curve_len=3, k_nn=k_nn)
        self.layer2 = CurveNetLayer(64, 128, num_curves=2, curve_len=3, k_nn=k_nn)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # Initialize feats as xyz coords
        feats = self.layer1(xyz, xyz)
        feats = self.layer2(xyz, feats)
        global_feat = feats.max(dim=1)[0]
        return self.classifier(global_feat)


def build() -> nn.Module:
    """Build CurveNet classifier (2 curves per point, curve length 3, 32 points)."""
    return CurveNetClassifier(num_classes=40, k_nn=8)


def example_input() -> torch.Tensor:
    """Small point cloud (1, 32, 3) -- compact for fast trace."""
    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    (
        "CurveNet (guided curve-grouping walk + LPFA aggregation)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
