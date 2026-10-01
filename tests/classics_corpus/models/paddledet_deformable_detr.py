"""PaddleDetection Deformable DETR with compact multi-scale deformable attention.

Zhu et al. (ICLR 2021), "Deformable DETR: Deformable Transformers for
End-to-End Object Detection".  Deformable DETR replaces dense attention over
all image tokens with attention over a small set of sampled locations around
reference points on multi-scale feature maps.

This package-free reconstruction keeps the multi-scale backbone, object queries,
reference-point offsets, sampled feature gathering, transformer-style FFN, and
class/box prediction heads.  Sampling is implemented with ``grid_sample`` so it
is traceable in the base PyTorch environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBackbone(nn.Module):
    """Small backbone returning two feature scales."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize the multi-scale backbone.

        Parameters
        ----------
        dim:
            Output feature dimension for each level.
        """

        super().__init__()
        self.level1 = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale features.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        list[torch.Tensor]
            Feature maps from two spatial scales.
        """

        level1 = self.level1(x)
        return [level1, self.level2(level1)]


class DeformableQueryAttention(nn.Module):
    """Query-conditioned multi-scale feature sampler."""

    def __init__(self, dim: int = 48, queries: int = 8, levels: int = 2, points: int = 4) -> None:
        """Initialize deformable query attention.

        Parameters
        ----------
        dim:
            Query and feature dimension.
        queries:
            Number of object queries.
        levels:
            Number of feature levels.
        points:
            Number of sampling points per level.
        """

        super().__init__()
        self.queries = queries
        self.levels = levels
        self.points = points
        self.offsets = nn.Linear(dim, levels * points * 2)
        self.weights = nn.Linear(dim, levels * points)
        self.value = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(levels)])
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        refs: torch.Tensor,
        feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Sample multi-scale features around query reference points.

        Parameters
        ----------
        query:
            Object query tensor of shape ``(batch, queries, dim)``.
        refs:
            Normalized reference points of shape ``(batch, queries, 2)``.
        feats:
            Multi-scale feature maps.

        Returns
        -------
        torch.Tensor
            Updated query tensor.
        """

        batch, queries, dim = query.shape
        offsets = torch.tanh(self.offsets(query)).view(batch, queries, self.levels, self.points, 2)
        weights = torch.softmax(
            self.weights(query).view(batch, queries, self.levels, self.points), dim=-1
        )
        gathered = query.new_zeros(batch, queries, dim)
        for level, feat in enumerate(feats):
            value = self.value[level](feat)
            grid = refs[:, :, None, :] + 0.15 * offsets[:, :, level]
            grid = grid.mul(2.0).sub(1.0).view(batch, queries * self.points, 1, 2)
            sampled = F.grid_sample(value, grid, align_corners=False)
            sampled = sampled.squeeze(-1).transpose(1, 2).view(batch, queries, self.points, dim)
            gathered = gathered + (sampled * weights[:, :, level, :, None]).sum(dim=2)
        return self.out(gathered)


class PaddleDetDeformableDETR(nn.Module):
    """Compact Deformable DETR detector."""

    def __init__(self, dim: int = 48, queries: int = 8, classes: int = 5) -> None:
        """Initialize Deformable DETR.

        Parameters
        ----------
        dim:
            Query and feature dimension.
        queries:
            Number of object queries.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.backbone = MultiScaleBackbone(dim=dim)
        self.query = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.ref_points = nn.Linear(dim, 2)
        self.attn = DeformableQueryAttention(dim=dim, queries=queries)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=False), nn.Linear(dim * 2, dim)
        )
        self.cls = nn.Linear(dim, classes)
        self.box = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict object classes and normalized boxes.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Query-level class logits and boxes.
        """

        feats = self.backbone(x)
        batch = x.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        refs = torch.sigmoid(self.ref_points(query))
        query = query + self.attn(query, refs, feats)
        query = query + self.ff(self.norm(query))
        return {"pred_logits": self.cls(query), "pred_boxes": torch.sigmoid(self.box(query))}


def build() -> nn.Module:
    """Build the compact PaddleDetection Deformable DETR model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetDeformableDETR().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_deformable_detr", "build", "example_input", "2021", "DC"),
]
