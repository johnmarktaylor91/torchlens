"""NetVLAD pooling layer, 2016, Relja Arandjelovic et al.

Paper: NetVLAD: CNN architecture for weakly supervised place recognition.
Local descriptors are softly assigned to learned centroids and residuals are
aggregated with intra-normalization and global L2 normalization.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [("NetVLAD Pooling Layer", "build", "example_input", "2016", "DC")]


class NetVLADPoolingLayer(nn.Module):
    """Traceable NetVLAD pooling and projection layer."""

    def __init__(self, channels: int = 512, clusters: int = 8, out_dim: int = 32) -> None:
        """Initialize soft-assignment and centroid parameters.

        Parameters
        ----------
        channels
            Descriptor channel count.
        clusters
            Number of VLAD clusters.
        out_dim
            Output projection dimension.
        """
        super().__init__()
        self.assignment = nn.Conv2d(channels, clusters, kernel_size=1)
        self.centroids = nn.Parameter(torch.randn(clusters, channels) * 0.1)
        self.proj = nn.Linear(clusters * channels, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Pool local descriptors with NetVLAD residual aggregation.

        Parameters
        ----------
        x
            Descriptor map with shape ``(B, 512, 14, 14)``.

        Returns
        -------
        Tensor
            Projected normalized VLAD descriptor.
        """
        assignment = torch.softmax(self.assignment(x), dim=1)
        residual = x.unsqueeze(1) - self.centroids.view(1, -1, x.shape[1], 1, 1)
        vlad = (assignment.unsqueeze(2) * residual).sum(dim=(-2, -1))
        vlad = F.normalize(vlad, dim=2)
        return self.proj(F.normalize(vlad.flatten(1), dim=1))


def build() -> nn.Module:
    """Build a compact NetVLAD layer.

    Returns
    -------
    nn.Module
        Random-initialized NetVLAD module.
    """
    return NetVLADPoolingLayer()


def example_input() -> Tensor:
    """Return a traceable descriptor map.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 512, 14, 14)``.
    """
    return torch.randn(1, 512, 14, 14)
