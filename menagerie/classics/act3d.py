"""Act3D compact random-init reconstruction.

Paper: 3D Feature Field Transformers for Multi-Task Robotic Manipulation
(Gervet, Xian, Gkanatsios, Fragkiadaki, CoRL 2023).

The distinctive mechanism is recurrent coarse-to-fine 3D point-grid sampling:
2D/multiview features are lifted to a 3D feature field, candidate action points
attend to that field with relative spatial attention, and the highest scoring
region recenters the next finer grid.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Act3DRelativeAttention(nn.Module):
    """Relative-position attention from candidate 3D action points to feature clouds."""

    def __init__(self, dim: int = 40) -> None:
        """Initialize projections for relative spatial attention."""

        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.out = nn.Linear(dim, dim)

    def forward(
        self, point_tokens: Tensor, grid_xyz: Tensor, field_tokens: Tensor, field_xyz: Tensor
    ) -> Tensor:
        """Featurize sampled grid points from the lifted 3D feature field."""

        rel = grid_xyz[:, :, None, :] - field_xyz[:, None, :, :]
        logits = torch.matmul(self.q(point_tokens), self.k(field_tokens).transpose(-1, -2))
        logits = logits / (point_tokens.shape[-1] ** 0.5) + self.bias(rel).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        return self.out(torch.matmul(weights, self.v(field_tokens)))


class Act3DPolicy(nn.Module):
    """Compact Act3D coarse-to-fine 6-DoF action detector."""

    def __init__(self, dim: int = 40, grid_points: int = 8) -> None:
        """Initialize feature lifting, tied attentions, and pose heads."""

        super().__init__()
        self.grid_points = grid_points
        self.rgbd_lift = nn.Linear(7, dim)
        self.lang = nn.Linear(12, dim)
        self.prop = nn.Linear(4, dim)
        self.grid_embed = nn.Linear(3, dim)
        self.attn = Act3DRelativeAttention(dim)
        self.score = nn.Linear(dim, 1)
        self.rot = nn.Linear(dim, 6)
        self.gripper = nn.Linear(dim, 1)

    def _grid(self, center: Tensor, radius: float) -> Tensor:
        """Create a small deterministic 3D candidate grid around a center."""

        offsets = torch.linspace(-radius, radius, self.grid_points, device=center.device)
        base = torch.stack([offsets, offsets.roll(1), offsets.roll(2)], dim=-1)
        return center[:, None, :] + base[None, :, :]

    def forward(
        self, inputs: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict translation logits, rotation, and gripper state from a 3D field."""

        xyz, rgbd_features, language, proprio = inputs
        field = self.rgbd_lift(torch.cat([xyz, rgbd_features], dim=-1))
        context = self.lang(language) + self.prop(proprio)
        center = xyz.mean(dim=1)
        coarse_xyz = self._grid(center, 1.0)
        coarse = self.grid_embed(coarse_xyz) + context[:, None, :]
        coarse = coarse + self.attn(coarse, coarse_xyz, field, xyz)
        coarse_scores = self.score(coarse).squeeze(-1)
        focus = torch.sum(torch.softmax(coarse_scores, dim=-1).unsqueeze(-1) * coarse_xyz, dim=1)
        fine_xyz = self._grid(focus, 0.25)
        fine = self.grid_embed(fine_xyz) + context[:, None, :]
        fine = fine + self.attn(fine, fine_xyz, field, xyz)
        fine_scores = self.score(fine).squeeze(-1)
        pooled = torch.sum(torch.softmax(fine_scores, dim=-1).unsqueeze(-1) * fine, dim=1)
        return fine_scores, self.rot(pooled), self.gripper(pooled)


def build() -> nn.Module:
    """Build a compact random-init Act3D policy."""

    return Act3DPolicy().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return lifted point coordinates, RGB-D features, language, and proprioception."""

    return (torch.randn(1, 16, 3), torch.randn(1, 16, 4), torch.randn(1, 12), torch.randn(1, 4))


MENAGERIE_ENTRIES = [
    ("Act3D", "build", "example_input", "2023", "DC"),
]
