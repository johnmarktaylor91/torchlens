"""Point2Vec: data2vec-style self-supervised representation learning on point clouds.

Point2Vec adapts data2vec to point clouds: FPS/kNN groups form local point
tokens, a masked student transformer predicts contextual targets produced by a
teacher transformer, and a projection loss is applied on masked point patches.
This compact random-init reconstruction keeps the architectural primitives
needed for TorchLens rendering.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Select point-patch centers with iterative farthest-point sampling.

    Parameters
    ----------
    xyz:
        Input point cloud, shape ``(batch, points, 3)``.
    npoint:
        Number of centers to select.

    Returns
    -------
    torch.Tensor
        Center indices with shape ``(batch, npoint)``.
    """

    batch, points, _ = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, points), 1e10, device=xyz.device)
    farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    batch_idx = torch.arange(batch, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        center = xyz[batch_idx, farthest].unsqueeze(1)
        dist = ((xyz - center) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1)[1]
    return centroids


def group_knn(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """Gather k-nearest relative-coordinate patches.

    Parameters
    ----------
    xyz:
        Input point cloud.
    centers:
        Patch centers.
    k:
        Number of neighbors per patch.

    Returns
    -------
    torch.Tensor
        Relative-coordinate groups, shape ``(batch, groups, k, 3)``.
    """

    distances = torch.cdist(centers, xyz)
    idx = distances.topk(k, dim=-1, largest=False)[1]
    batch_idx = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1)
    grouped = xyz[batch_idx, idx]
    return grouped - centers.unsqueeze(2)


class PointPatchTokenizer(nn.Module):
    """FPS/kNN point-patch tokenizer with shared PointNet pooling."""

    def __init__(self, groups: int = 8, neighbors: int = 4, dim: int = 48) -> None:
        """Initialize tokenizer layers.

        Parameters
        ----------
        groups:
            Number of point patches.
        neighbors:
            Number of neighbors in each patch.
        dim:
            Token width.
        """

        super().__init__()
        self.groups = groups
        self.neighbors = neighbors
        self.dim = dim
        self.local = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a point cloud into contextual patch tokens.

        Parameters
        ----------
        xyz:
            Input point cloud.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Patch tokens and patch centers.
        """

        center_idx = farthest_point_sample(xyz, self.groups)
        batch_idx = torch.arange(xyz.shape[0], device=xyz.device).unsqueeze(1)
        centers = xyz[batch_idx, center_idx]
        rel = group_knn(xyz, centers, self.neighbors)
        local = self.local(rel.reshape(-1, 3)).reshape(
            xyz.shape[0], self.groups, self.neighbors, self.dim
        )
        tokens = local.max(dim=2)[0] + self.pos(centers)
        return tokens, centers


class Point2Vec(nn.Module):
    """Masked student and teacher transformer for point-cloud data2vec."""

    def __init__(self, dim: int = 48, groups: int = 8, mask_ratio: float = 0.5) -> None:
        """Initialize Point2Vec.

        Parameters
        ----------
        dim:
            Transformer width.
        groups:
            Number of point patches.
        mask_ratio:
            Fraction of patches replaced by the mask token.
        """

        super().__init__()
        self.groups = groups
        self.mask_ratio = mask_ratio
        self.tokenizer = PointPatchTokenizer(groups=groups, dim=dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=96, dropout=0.0, batch_first=True
        )
        self.student = nn.TransformerEncoder(layer, num_layers=2)
        teacher_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=96, dropout=0.0, batch_first=True
        )
        self.teacher = nn.TransformerEncoder(teacher_layer, num_layers=2)
        self.predictor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict teacher contextual targets for masked point patches.

        Parameters
        ----------
        xyz:
            Input point cloud.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Student predictions and detached teacher targets for masked patches.
        """

        tokens, _ = self.tokenizer(xyz)
        mask_count = int(self.groups * self.mask_ratio)
        visible_count = self.groups - mask_count
        teacher_targets = self.teacher(tokens).detach()
        mask_tokens = self.mask_token.expand(tokens.shape[0], mask_count, -1)
        student_in = torch.cat([tokens[:, :visible_count], mask_tokens], dim=1)
        student_out = self.student(student_in)
        pred = self.predictor(student_out[:, visible_count:])
        target = teacher_targets[:, visible_count:]
        return pred, target


def build() -> nn.Module:
    """Build compact Point2Vec.

    Returns
    -------
    nn.Module
        Random-initialized Point2Vec masked self-supervised model.
    """

    return Point2Vec().eval()


def example_input() -> torch.Tensor:
    """Create a compact point cloud.

    Returns
    -------
    torch.Tensor
        Point cloud with shape ``(1, 32, 3)``.
    """

    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    (
        "Point2Vec",
        "build",
        "example_input",
        "2023",
        "DC",
    ),
]
