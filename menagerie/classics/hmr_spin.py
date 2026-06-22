"""HMR and SPIN: Human Mesh Recovery with Iterative Error Feedback.

HMR (Kanazawa et al., 2018):
  Paper: https://arxiv.org/abs/1712.06584
  Source: https://github.com/akanazawa/hmr

SPIN (Kolotouros et al., 2019):
  Paper: https://arxiv.org/abs/1909.12828
  Source: https://github.com/nkolot/SPIN

Architecture:
  Both share the same forward architecture:
    1. ResNet-50 image encoder -> global average pool -> 2048-d feature vector.
    2. Iterative Error Feedback (IEF) regressor:
       In each of T iterations (paper: T=3):
         concat([image_features, current_theta]) -> fc(2048+85 -> 1024) -> fc(1024 -> 85)
         delta_theta += fc_out; theta = init_theta + delta_theta
       theta encodes SMPL pose (72-d), shape (10-d), weak-perspective camera (3-d) = 85-d.

  SPIN's training differs (in-the-loop SMPLify fitting optimization for pseudo-GT),
  but the forward architecture is identical to HMR.  The SMPL mesh generation layer
  (Linear Blend Skinning) is outside the scope of the network forward pass for
  trace purposes and is documented as omitted.

Faithful compact simplification:
  ResNet-50 backbone replaced by a compact 4-stage CNN with same output dim (2048-d
  after global average pool).  The IEF regressor (2 FC layers, T=3 iterations) is
  reproduced exactly.  Trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# SMPL parameter vector size: 72 pose + 10 shape + 3 camera = 85
SMPL_DIM = 85


# ---------------------------------------------------------------------------
# Compact ResNet-50-style encoder (same output channels as ResNet-50)
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class CompactResNet50Encoder(nn.Module):
    """Compact CNN encoder producing 2048-d features (matching ResNet-50 pooled output)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock(64, 256, 1))
        self.layer2 = nn.Sequential(_ResBlock(256, 512, 2))
        self.layer3 = nn.Sequential(_ResBlock(512, 1024, 2))
        self.layer4 = nn.Sequential(_ResBlock(1024, 2048, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 2048)


# ---------------------------------------------------------------------------
# IEF Regressor
# ---------------------------------------------------------------------------


class IEFRegressor(nn.Module):
    """Iterative Error Feedback regressor (HMR/SPIN).

    Args:
        feat_dim:  Image feature dimension (2048).
        smpl_dim:  SMPL parameter dimension (85).
        n_iter:    Number of IEF iterations (paper: 3).
    """

    def __init__(self, feat_dim: int = 2048, smpl_dim: int = SMPL_DIM, n_iter: int = 3) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.smpl_dim = smpl_dim
        in_dim = feat_dim + smpl_dim
        self.fc1 = nn.Linear(in_dim, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, smpl_dim)
        # Init: mean SMPL params (zeros here; paper uses pre-fit mean)
        self.register_buffer("init_theta", torch.zeros(1, smpl_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.size(0)
        theta = self.init_theta.expand(B, -1)
        for _ in range(self.n_iter):
            xc = torch.cat([features, theta], dim=1)
            h = self.drop1(F.relu(self.fc1(xc)))
            h = self.drop2(F.relu(self.fc2(h)))
            delta = self.decpose(h)
            theta = theta + delta
        return theta  # (B, 85)


# ---------------------------------------------------------------------------
# Full HMR / SPIN model
# ---------------------------------------------------------------------------


class HMRNet(nn.Module):
    """HMR / SPIN: ResNet encoder + IEF regressor -> SMPL theta (pose/shape/camera)."""

    def __init__(self, n_iter: int = 3) -> None:
        super().__init__()
        self.encoder = CompactResNet50Encoder()
        self.regressor = IEFRegressor(feat_dim=2048, smpl_dim=SMPL_DIM, n_iter=n_iter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        theta = self.regressor(features)
        return theta  # (B, 85): [pose(72), shape(10), camera(3)]


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_hmr_resnet50() -> nn.Module:
    """Build HMR (Human Mesh Recovery, Kanazawa 2018) with compact ResNet-50 encoder."""
    return HMRNet(n_iter=3)


def build_spin_resnet50() -> nn.Module:
    """Build SPIN (Kolotouros 2019) -- same forward arch as HMR, SPIN trains with in-loop SMPLify."""
    return HMRNet(n_iter=3)


def example_input_hmr() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "HMR-ResNet50 (human mesh recovery, iterative error feedback)",
        "build_hmr_resnet50",
        "example_input_hmr",
        "2018",
        "DC",
    ),
    (
        "SPIN-ResNet50 (in-the-loop SMPLify, same IEF arch as HMR)",
        "build_spin_resnet50",
        "example_input_hmr",
        "2019",
        "DC",
    ),
]
