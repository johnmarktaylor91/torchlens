"""Vector Neurons (VNN): a general framework for SO(3)-equivariant networks.

Deng et al., ICCV 2021.
Paper: https://arxiv.org/abs/2104.12229
Source: https://github.com/FlyingGiraffe/vnn

Distinctive primitive -- LIST every neuron is a 3D VECTOR, not a scalar:
  A VNN feature is a tensor of shape (B, C, 3, N): each of the C channels is a
  3-vector (per point N). All layers are built to commute with a rotation R in
  SO(3) applied to that 3-axis, so the whole network is rotation-EQUIVARIANT.

  - VNLinear: a plain channel mixing  W @ x  applied along the channel axis;
    because the 3-axis is untouched, it commutes with rotation.
  - VNLeakyReLU (VN nonlinearity): cannot threshold a vector elementwise (that
    breaks equivariance). Instead it learns a direction k per channel, and for
    each vector q it keeps the component along k if q.k >= 0, else it REMOVES
    the part of q pointing against k (reflects/clips along the learned plane).
    This is equivariant because k rotates with the input.
  - VNStdFeature / invariant readout: project features onto a learned, input-
    derived rotation-equivariant frame, yielding rotation-INVARIANT scalars for
    a classification/regression head.

This module reproduces VNLinear + VNLeakyReLU + a VN invariant readout stacked
into a small VN-PointNet-style encoder. Faithful compact random-init reimpl;
small channel counts and few points so the unrolled trace draws quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VNLinear(nn.Module):
    """Rotation-equivariant linear: mixes channels, leaves the 3-axis untouched."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.map = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, 3, N) -> (B, C_out, 3, N)
        return self.map(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    """VN nonlinearity: clip each vector against a learned per-channel direction."""

    def __init__(self, in_ch: int, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.dir = VNLinear(in_ch, in_ch)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3, N)
        k = self.dir(x)  # learned direction per channel (B, C, 3, N)
        dotprod = (x * k).sum(dim=2, keepdim=True)  # (B, C, 1, N)
        k_norm_sq = (k * k).sum(dim=2, keepdim=True) + 1e-6
        # component of x along k
        proj = (dotprod / k_norm_sq) * k
        mask = (dotprod >= 0).float()
        # keep x where aligned, else remove the against-k component (leaky)
        out = mask * x + (1.0 - mask) * (
            self.negative_slope * x + (1.0 - self.negative_slope) * (x - proj)
        )
        return out


class VNBatchNorm(nn.Module):
    """Equivariant norm: scale each vector by a learned function of its magnitude."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize by the (invariant) magnitude, then re-apply direction
        norm = torch.norm(x, dim=2) + 1e-6  # (B, C, N)
        norm_bn = self.bn(norm)
        return x * (norm_bn / norm).unsqueeze(2)


class VNStdFeature(nn.Module):
    """Invariant readout: build a learned equivariant frame, project to invariant scalars."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.vn1 = VNLinear(in_ch, in_ch // 2)
        self.vn2 = VNLinear(in_ch // 2, 3)  # 3 vectors define a rotation-equivariant frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3, N)
        z = self.vn2(VNLeakyReLU(x.shape[1] // 2)(self.vn1(x)))  # (B, 3, 3, N) frame
        # project features onto the frame -> invariant (B, C, 3, N) scalars
        # einsum over the 3-axis: x_inv[b,c,j,n] = sum_i x[b,c,i,n] * z[b,j,i,n]
        x_inv = torch.einsum("bcin,bjin->bcjn", x, z)
        return x_inv


class VNPointNetEncoder(nn.Module):
    """Small VN-PointNet: VNLinear/VNLeakyReLU stack -> VN invariant head."""

    def __init__(self, n_classes: int = 10, ch: int = 16) -> None:
        super().__init__()
        self.vn1 = VNLinear(1, ch)
        self.act1 = VNLeakyReLU(ch)
        self.vn2 = VNLinear(ch, ch * 2)
        self.act2 = VNLeakyReLU(ch * 2)
        self.bn = VNBatchNorm(ch * 2)
        self.std = VNStdFeature(ch * 2)
        self.head = nn.Sequential(
            nn.Linear((ch * 2) * 3, 64), nn.ReLU(inplace=True), nn.Linear(64, n_classes)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, 3, N) point cloud -> lift to (B, 1, 3, N)
        x = pts.unsqueeze(1)
        x = self.act1(self.vn1(x))
        x = self.act2(self.vn2(x))
        x = self.bn(x)
        x_inv = self.std(x)  # (B, C, 3, N) invariant
        feat = x_inv.mean(dim=-1).flatten(1)  # pool over points
        return self.head(feat)


def build_vnn() -> nn.Module:
    """Build a compact VN-PointNet (SO(3)-equivariant Vector Neurons encoder)."""
    return VNPointNetEncoder(n_classes=10, ch=16)


def example_input() -> torch.Tensor:
    """Example point cloud ``(1, 3, 256)`` (B, xyz, N) for Vector Neurons."""
    return torch.randn(1, 3, 256)


MENAGERIE_ENTRIES = [
    (
        "Vector Neurons VN-PointNet (SO(3)-equivariant 3D-vector neurons)",
        "build_vnn",
        "example_input",
        "2021",
        "DC",
    ),
]
