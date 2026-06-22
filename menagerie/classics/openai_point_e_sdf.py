"""OpenAI Point-E SDF regressor for point-cloud-to-mesh conversion.

Paper: Nichol et al. 2022, "Point-E: A System for Generating 3D Point Clouds
from Complex Prompts"; official model card describes SDF regression models.

This compact module represents the released SDF stage: query coordinates attend
to point-cloud features, then an MLP predicts signed distance values.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointESDF(nn.Module):
    """Compact point-conditioned signed-distance regressor."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize point/query encoders and SDF head.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.point_in = nn.Linear(3, dim)
        self.query_in = nn.Linear(3, dim)
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.sdf = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, points: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """Predict signed distances at query coordinates.

        Parameters
        ----------
        points:
            Conditioning point cloud.
        queries:
            Query coordinates.

        Returns
        -------
        torch.Tensor
            Signed-distance predictions.
        """

        point_tokens = self.point_in(points)
        query_tokens = self.query_in(queries)
        attn, _ = self.cross(query_tokens, point_tokens, point_tokens)
        return self.sdf(query_tokens + attn)


def build() -> nn.Module:
    """Build the compact Point-E SDF model.

    Returns
    -------
    nn.Module
        Random-initialized SDF regressor.
    """

    return PointESDF()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create conditioning points and SDF queries.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Point cloud and query coordinates.
    """

    return torch.randn(1, 12, 3), torch.randn(1, 10, 3)


MENAGERIE_ENTRIES = [
    ("openai_point_e_sdf", "build", "example_input", "2022", "E6"),
]
