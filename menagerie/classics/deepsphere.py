"""DeepSphere compact graph-based spherical CNN.

Perraudin et al. and Defferrard et al., "DeepSphere: a graph-based spherical
CNN", use a graph Laplacian on a sampled sphere and learn isotropic filters as
Chebyshev polynomials of that Laplacian.  Pooling coarsens the spherical graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution over a fixed spherical graph Laplacian."""

    def __init__(
        self, in_channels: int, out_channels: int, order: int, laplacian: torch.Tensor
    ) -> None:
        """Initialize Chebyshev filter weights.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        order:
            Chebyshev polynomial order.
        laplacian:
            Scaled graph Laplacian.
        """

        super().__init__()
        self.order = order
        self.weight = nn.Parameter(torch.randn(order, in_channels, out_channels) * 0.05)
        self.register_buffer("laplacian", laplacian)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Chebyshev polynomial graph filtering.

        Parameters
        ----------
        x:
            Graph signal ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Filtered graph signal.
        """

        terms = [x, torch.einsum("nm,bmc->bnc", self.laplacian, x)]
        for _ in range(2, self.order):
            terms.append(2.0 * torch.einsum("nm,bmc->bnc", self.laplacian, terms[-1]) - terms[-2])
        basis = torch.stack(terms[: self.order], dim=2)
        return torch.einsum("bnkc,kco->bno", basis, self.weight)


class DeepSphereCompact(nn.Module):
    """Small DeepSphere classifier on an eight-neighbor ring-like spherical graph."""

    def __init__(self, nodes: int = 16) -> None:
        """Initialize graph filters and classifier.

        Parameters
        ----------
        nodes:
            Number of sampled sphere vertices.
        """

        super().__init__()
        adj = torch.zeros(nodes, nodes)
        for i in range(nodes):
            adj[i, (i - 1) % nodes] = 1.0
            adj[i, (i + 1) % nodes] = 1.0
            adj[i, (i + nodes // 2) % nodes] = 0.5
        deg = adj.sum(dim=1)
        lap = torch.eye(nodes) - adj / deg.view(-1, 1).clamp_min(1.0)
        self.conv1 = ChebGraphConv(3, 16, 4, lap)
        self.conv2 = ChebGraphConv(16, 24, 4, lap)
        self.head = nn.Linear(24, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a spherical graph signal.

        Parameters
        ----------
        x:
            Graph signal ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.head(x.mean(dim=1))


def build() -> nn.Module:
    """Build compact DeepSphere.

    Returns
    -------
    nn.Module
        Spherical graph CNN.
    """

    return DeepSphereCompact()


def example_input() -> torch.Tensor:
    """Create a spherical graph signal.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 16, 3)``.
    """

    return torch.randn(1, 16, 3)


MENAGERIE_ENTRIES = [
    ("DeepSphere (Chebyshev graph CNN on sampled sphere)", "build", "example_input", "2020", "DC"),
]
