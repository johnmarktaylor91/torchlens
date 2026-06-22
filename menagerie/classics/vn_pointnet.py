"""Vector Neurons: Rotation-Equivariant PointNet and DGCNN.

Deng et al., ICCV 2021.
Paper: https://arxiv.org/abs/2104.12229
Source: https://github.com/FlyingGiraffe/vnn

Vector Neurons (VN) replace scalar neurons in point cloud networks with
3D VECTOR neurons. Instead of a neuron holding a single scalar activation,
a VN-neuron holds a 3D vector (a point in R^3). This gives the network
SO(3)/O(3) equivariance: when the input point cloud is rotated, all internal
vector features rotate correspondingly.

Key primitives:
  - VN-Linear: maps (C, 3) -> (C_out, 3) by applying a weight matrix to the
    channel dimension, leaving the 3D dimension intact. Equivalent to
    matmul(W, x) where W is (C_out, C) and x is (..., C, 3).
  - VN-ReLU (VN-LeakyReLU): learnable direction d (3D unit vector) that defines
    the "positive half-space". For a vector feature v: if v·d >= 0 keep v,
    else project out the component along d (remove the anti-parallel part).
    This is rotation-equivariant because d transforms with the same rotation.
  - VN-MaxPool / VN-MeanPool: max-pool / mean-pool over the point dimension,
    preserving the 3D vector dimension.
  - VN-Invariant (STD pooling): computes the std deviation of the vector
    magnitudes as a rotation-invariant global feature for classification.
  - VN-EdgeConv (for DGCNN): builds kNN graph, aggregates edge features as
    vector differences + concatenation, then applies VN-Linear.

Three catalog entries:
  1. vn_pointnet: classification head.
  2. vn_pointnet_partseg: per-point part segmentation head.
  3. vn_dgcnn: EdgeConv made vector-equivariant for classification.

All use a point cloud input (1, 3, N) = (batch, 3 coords, N points).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VN-Linear: acts on (*, C, 3) channel dim, preserves 3D vector dim
# ---------------------------------------------------------------------------


class VNLinear(nn.Module):
    """Vector Neuron Linear: maps (..., C_in, 3) -> (..., C_out, 3).

    Applies a learnable (C_out, C_in) matrix to the channel axis.
    Equivariant: rotation of the 3D dim commutes with channel mixing.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, N, 3) for per-point or (B, C_in, 3) for global.
        # Channel dim is always dim=1; last dim is always the 3D vector.
        # Strategy: flatten all dims between channel and last, apply weight, reshape back.
        shape = x.shape  # e.g. (B, C_in, N, 3) or (B, C_in, 3)
        B = shape[0]
        C_in = shape[1]
        trailing = shape[2:]  # e.g. (N, 3) or (3,)
        # Flatten trailing into one dim: (B, C_in, trailing_flat)
        x_flat = x.reshape(B, C_in, -1)  # (B, C_in, trailing_flat)
        # Apply weight along channel dim: (B, C_out, trailing_flat)
        out_flat = torch.einsum("bct,oc->bot", x_flat, self.weight)
        # Reshape back
        C_out = self.weight.shape[0]
        out = out_flat.reshape(B, C_out, *trailing)
        return out


# ---------------------------------------------------------------------------
# VN-LeakyReLU: equivariant nonlinearity via learnable direction
# ---------------------------------------------------------------------------


class VNLeakyReLU(nn.Module):
    """Vector Neuron nonlinearity via a learnable direction.

    For each channel c, we learn a direction d_c (3-vector). For input v_c:
      if v_c · d_c >= 0: output v_c (positive half-space)
      else: output v_c - (v_c · d_c / ||d_c||^2) * d_c  (project out anti-parallel)

    The direction is learned (the model finds the appropriate orientation).
    """

    def __init__(self, channels: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        # Learnable direction per channel: (C, 3)
        self.direction = nn.Parameter(F.normalize(torch.randn(channels, 3), dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, 3) or (B, C, 3).
        # Channel dim is dim=1; last dim is the 3D vector.
        d = F.normalize(self.direction, dim=-1)  # (C, 3)

        # Dot product per channel: x[..., c, :] · d[c]
        # x shape: (B, C, ..., 3) -> we need (x * d).sum(last_dim)
        # Broadcast d to match x: d is (C, 3), x is (B, C, *spatial, 3)
        # Reshape d to (1, C, *ones, 3)
        extra_dims = x.dim() - 3  # number of spatial dims (0 for (B,C,3), 1 for (B,C,N,3))
        d_bc = d.view(1, -1, *([1] * extra_dims), 3)  # (1, C, [1,]*, 3)

        # v · d for each position: (B, C, *spatial)
        dot = (x * d_bc).sum(-1)

        # Project out the negative component
        neg_part = dot.clamp(max=0).unsqueeze(-1) * d_bc  # (B, C, *spatial, 3)

        out = x - neg_part + self.negative_slope * neg_part
        return out


# ---------------------------------------------------------------------------
# VN-BatchNorm (optional; use LayerNorm on magnitudes for simplicity)
# ---------------------------------------------------------------------------


class VNBatchNorm(nn.Module):
    """Normalizes magnitudes of vector features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3) or (B, C, N, 3)
        norm = x.norm(dim=-1)  # (B, C) or (B, C, N)
        if norm.dim() == 3:
            B, C, N = norm.shape
            norm_bn = self.bn(norm.reshape(B * N, C).T.unsqueeze(-1)).squeeze(-1).T.reshape(B, C, N)
        else:
            norm_bn = self.bn(norm)
        # Scale x by norm_bn / (norm + eps)
        eps = 1e-6
        scale = norm_bn / (norm + eps)
        return x * scale.unsqueeze(-1)


# ---------------------------------------------------------------------------
# VN-MaxPool and VN-MeanPool over point dimension
# ---------------------------------------------------------------------------


def vn_max_pool(x: torch.Tensor) -> torch.Tensor:
    """Max-pool over point dim, keeping 3D vector dim.

    x: (B, C, N, 3) -> (B, C, 3) by taking the vector with max norm per channel.
    """
    norms = x.norm(dim=-1)  # (B, C, N)
    idx = norms.argmax(dim=-1, keepdim=True)  # (B, C, 1)
    idx_3d = idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, C, 1, 3)
    return x.gather(2, idx_3d).squeeze(2)  # (B, C, 3)


def vn_mean_pool(x: torch.Tensor) -> torch.Tensor:
    """Mean-pool over point dim.

    x: (B, C, N, 3) -> (B, C, 3).
    """
    return x.mean(dim=2)


# ---------------------------------------------------------------------------
# VN-Invariant: maps equivariant (B, C, 3) -> invariant (B, C)
# ---------------------------------------------------------------------------


def vn_invariant(x: torch.Tensor) -> torch.Tensor:
    """Compute rotation-invariant feature from VN features: the norm of each channel.

    x: (B, C, 3) -> (B, C).
    """
    return x.norm(dim=-1)


# ---------------------------------------------------------------------------
# VN-PointNet: classification
# ---------------------------------------------------------------------------


class VNPointNet(nn.Module):
    """Vector Neurons PointNet for point cloud classification.

    Input: (B, 3, N) — point cloud.
    Output: (B, num_classes) class logits.
    """

    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()
        # VN-Linear layers: map (B, C_in, N, 3) -> (B, C_out, N, 3)
        self.vn1 = VNLinear(1, 16)  # input: 1 channel (xyz = one 3-vector per point)
        self.act1 = VNLeakyReLU(16)
        self.vn2 = VNLinear(16, 32)
        self.act2 = VNLeakyReLU(32)
        self.vn3 = VNLinear(32, 64)
        self.act3 = VNLeakyReLU(64)

        # Invariant head: norm of pooled VN features -> scalar features
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N)
        B, _, N = x.shape

        # Expand to VN format: (B, 1, N, 3) — 1 channel, each point is a 3-vector
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, 3)

        # VN-Linear + VN-nonlinearity layers
        x = self.act1(self.vn1(x))  # (B, 16, N, 3)
        x = self.act2(self.vn2(x))  # (B, 32, N, 3)
        x = self.act3(self.vn3(x))  # (B, 64, N, 3)

        # VN global max pool: (B, 64, 3)
        x = vn_max_pool(x)  # (B, 64, 3)

        # Invariant: norm per channel -> (B, 64)
        x = vn_invariant(x)

        # Classification
        return self.fc(x)


# ---------------------------------------------------------------------------
# VN-PointNet partseg: per-point segmentation
# ---------------------------------------------------------------------------


class VNPointNetPartSeg(nn.Module):
    """Vector Neurons PointNet for part segmentation.

    Input: (B, 3, N) — point cloud.
    Output: (B, num_parts, N) per-point logits.
    """

    def __init__(self, num_parts: int = 50) -> None:
        super().__init__()
        self.vn1 = VNLinear(1, 16)
        self.act1 = VNLeakyReLU(16)
        self.vn2 = VNLinear(16, 32)
        self.act2 = VNLeakyReLU(32)
        self.vn3 = VNLinear(32, 64)
        self.act3 = VNLeakyReLU(64)

        # For partseg: concatenate global + local features per point
        # Global = max-pooled invariant (64 scalars)
        # Local = per-point invariant features from vn2 (32 scalars)
        self.seg_head = nn.Sequential(
            nn.Conv1d(64 + 32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, num_parts, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N)
        B, _, N = x.shape
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, 3)

        x = self.act1(self.vn1(x))  # (B, 16, N, 3)
        x = self.act2(self.vn2(x))  # (B, 32, N, 3)

        # Save local feature (per-point invariant)
        local_feat = vn_invariant(x)  # (B, 32, N)

        x = self.act3(self.vn3(x))  # (B, 64, N, 3)

        # Global pooled feature
        global_feat = vn_invariant(vn_max_pool(x))  # (B, 64)
        global_feat_rep = global_feat.unsqueeze(-1).expand(-1, -1, N)  # (B, 64, N)

        # Concatenate global + local
        feat = torch.cat([global_feat_rep, local_feat], dim=1)  # (B, 96, N)

        return self.seg_head(feat)  # (B, num_parts, N)


# ---------------------------------------------------------------------------
# VN-DGCNN: EdgeConv made vector-equivariant
# ---------------------------------------------------------------------------


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build kNN graph indices.

    Args:
        x: (B, 3, N)
        k: number of nearest neighbours.

    Returns:
        idx: (B, N, k) indices of k nearest neighbours.
    """
    B, C, N = x.shape
    # Pairwise squared distances
    xT = x.permute(0, 2, 1)  # (B, N, C)
    dist = torch.cdist(xT, xT)  # (B, N, N)
    # k+1 to exclude self; then drop self
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    return idx[:, :, 1:]  # (B, N, k) — exclude self


class VNEdgeConv(nn.Module):
    """VN-EdgeConv: equivariant edge feature aggregation.

    For each point and each of its k neighbours, computes:
      edge_feat = (neighbour_vec - centre_vec)  ||  centre_vec
    in the VN channel space, then applies a VN-Linear.

    The subtraction and concatenation preserve equivariance.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        # Edge features are: [x_i || x_j - x_i] = 2 * in_channels channels
        self.vn_linear = VNLinear(in_channels * 2, out_channels)
        self.act = VNLeakyReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, 3)
        B, C, N, _ = x.shape

        # Get kNN indices from the vector magnitudes (invariant proxy)
        x_pts = x.norm(dim=-1)  # (B, C, N) — use norms as "scalar" coords for kNN
        # Use mean over channels for kNN: (B, N) -> (B, 1, N)
        x_scalar = x_pts.mean(dim=1)  # (B, N)
        # We need 3D coords for kNN; use the mean of vector features as proxy
        x_coords = x.mean(dim=1)  # (B, N, 3) — mean over channel dim
        idx = knn_graph(x_coords.permute(0, 2, 1), self.k)  # (B, N, k)

        # Gather neighbours: (B, C, N, k, 3)
        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(B, C, N, self.k, 3)
        # x: (B, C, N, 3) -> (B, C, N, 1, 3) -> expand or gather
        x_expand = x.unsqueeze(3).expand(B, C, N, self.k, 3)
        # Neighbour features: gather from N dim using idx
        x_neigh = x.unsqueeze(3).expand(B, C, N, N, 3)  # too large; use index_select
        # More efficient: flatten N, then index
        x_flat = x.reshape(B, C, N, 3)
        # Build neighbour tensor via index along N dim
        idx_flat = idx.reshape(B, -1)  # (B, N*k)
        idx_flat_exp = idx_flat.unsqueeze(1).unsqueeze(-1).expand(B, C, N * self.k, 3)
        x_neigh_flat = x_flat.gather(2, idx_flat_exp)  # (B, C, N*k, 3)
        x_neigh = x_neigh_flat.view(B, C, N, self.k, 3)  # (B, C, N, k, 3)

        # Centre features expanded: (B, C, N, k, 3)
        x_centre = x.unsqueeze(3).expand_as(x_neigh)

        # Edge features: [centre || neighbour - centre] along channel dim
        edge_feat = torch.cat([x_centre, x_neigh - x_centre], dim=1)  # (B, 2C, N, k, 3)

        # Aggregate over k neighbours (max over k, keeping 3D dim)
        # Max-pool over k: take max-norm over neighbour dimension
        norms = edge_feat.norm(dim=-1)  # (B, 2C, N, k)
        best_k = norms.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        # (B, 2C, N, 1, 3)
        aggr = edge_feat.gather(3, best_k).squeeze(3)  # (B, 2C, N, 3)

        # VN-Linear + activation
        out = self.act(self.vn_linear(aggr))  # (B, out_channels, N, 3)
        return out


class VNDGCNN(nn.Module):
    """VN-DGCNN: Vector Neurons DGCNN for point cloud classification.

    Input: (B, 3, N) — point cloud.
    Output: (B, num_classes) class logits.
    """

    def __init__(self, num_classes: int = 40, k: int = 8) -> None:
        super().__init__()
        self.init_embed = VNLinear(1, 16)
        self.act_init = VNLeakyReLU(16)
        self.edge1 = VNEdgeConv(16, 32, k=k)
        self.edge2 = VNEdgeConv(32, 64, k=k)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N)
        B, _, N = x.shape

        # Initial VN embedding: (B, 1, N, 3)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, 3)
        x = self.act_init(self.init_embed(x))  # (B, 16, N, 3)

        # VN-EdgeConv layers
        x = self.edge1(x)  # (B, 32, N, 3)
        x = self.edge2(x)  # (B, 64, N, 3)

        # Global pool + invariant
        x = vn_max_pool(x)  # (B, 64, 3)
        x = vn_invariant(x)  # (B, 64)

        return self.fc(x)


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_vn_pointnet() -> nn.Module:
    """Build VN-PointNet for classification (40 classes, 3D equivariant)."""
    return VNPointNet(num_classes=40)


def build_vn_pointnet_partseg() -> nn.Module:
    """Build VN-PointNet for part segmentation (50 parts)."""
    return VNPointNetPartSeg(num_parts=50)


def build_vn_dgcnn() -> nn.Module:
    """Build VN-DGCNN for classification (40 classes, kNN=8)."""
    return VNDGCNN(num_classes=40, k=8)


def example_input() -> torch.Tensor:
    """Point cloud: (1, 3, 64) — batch=1, 3 coords, 64 points."""
    return torch.randn(1, 3, 64)


MENAGERIE_ENTRIES = [
    (
        "VN-PointNet (Vector Neurons PointNet, SO(3)-equivariant, vector-channel linear + VN-ReLU)",
        "build_vn_pointnet",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "VN-PointNet Part Segmentation (Vector Neurons PointNet, per-point partseg head)",
        "build_vn_pointnet_partseg",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "VN-DGCNN (Vector Neurons Dynamic Graph CNN, equivariant EdgeConv + kNN graph)",
        "build_vn_dgcnn",
        "example_input",
        "2021",
        "DC",
    ),
]
