"""Occupancy Network with a compact ResNet-18-style image encoder.

Paper: Mescheder et al., "Occupancy Networks: Learning 3D Reconstruction in
Function Space", CVPR 2019.

The original single-image setting uses a ResNet-18 encoder and a conditional
batch-normalized ResNet MLP decoder over 3D query points.  This version keeps
that encoder/implicit-decoder structure with reduced widths.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Small ResNet basic block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize residual convolutions."""

        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual image block."""

        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + self.skip(x))


class ConditionalResBlock(nn.Module):
    """Conditional batch-normalized decoder residual block."""

    def __init__(self, hidden: int, cond_dim: int) -> None:
        """Initialize conditional affine and MLP layers."""

        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.gamma = nn.Linear(cond_dim, hidden)
        self.beta = nn.Linear(cond_dim, hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Update point features conditioned on an image code."""

        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        z = (x - mean) / std
        z = z * (1.0 + self.gamma(cond).unsqueeze(1)) + self.beta(cond).unsqueeze(1)
        z = F.relu(self.fc1(z))
        return x + self.fc2(z)


class OccupancyNetwork(nn.Module):
    """Compact image-conditioned implicit occupancy model."""

    def __init__(self, cond_dim: int = 64, hidden: int = 64) -> None:
        """Initialize ResNet-style encoder and implicit decoder."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            BasicBlock(16, 16),
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, cond_dim),
        )
        self.point = nn.Linear(3, hidden)
        self.blocks = nn.ModuleList([ConditionalResBlock(hidden, cond_dim) for _ in range(4)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict occupancy logits for query points."""

        image, points = data
        cond = self.encoder(image)
        x = self.point(points)
        for block in self.blocks:
            x = block(x, cond)
        return self.head(F.relu(x)).squeeze(-1)


def build() -> nn.Module:
    """Build compact Occupancy Network."""

    return OccupancyNetwork()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return image and 3D query points."""

    return torch.randn(1, 3, 32, 32), torch.randn(1, 24, 3)


MENAGERIE_ENTRIES = [
    ("occupancy_network_resnet18", "build", "example_input", "2019", "geometry"),
]
