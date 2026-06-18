"""Growing Neural Gas, 1995, Fritzke.

Paper: Fritzke 1995, "A growing neural gas network learns topologies." The
original online algorithm inserts nodes and prunes aged edges; this traceable
forward pass exposes the core winner-neighbor adaptation signal without mutating
topology during inference.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Growing Neural Gas (GNG)", "build", "example_input", "1995", "CF")]


class GrowingNeuralGas(nn.Module):
    """Small neural-gas topology with differentiable soft adaptation scores."""

    def __init__(self, dim: int = 4, n_nodes: int = 8) -> None:
        """Initialize node positions and a chain adjacency.

        Parameters
        ----------
        dim
            Input dimensionality.
        n_nodes
            Number of current graph nodes.
        """
        super().__init__()
        self.nodes = nn.Parameter(torch.randn(n_nodes, dim))
        adjacency = torch.eye(n_nodes)
        adjacency = adjacency + torch.diag(torch.ones(n_nodes - 1), diagonal=1)
        adjacency = adjacency + torch.diag(torch.ones(n_nodes - 1), diagonal=-1)
        self.register_buffer("adjacency", adjacency.clamp_max(1.0))

    def forward(self, x: Tensor) -> Tensor:
        """Compute topology-aware soft winner responses.

        Parameters
        ----------
        x
            Input samples of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Concatenated winner, neighbor, and quantization-error features.
        """
        dist = torch.cdist(x, self.nodes)
        winner = torch.softmax(-dist * 8.0, dim=-1)
        neighbor = winner @ self.adjacency
        neighbor = neighbor / neighbor.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        reconstruction = neighbor @ self.nodes
        error = ((x - reconstruction) ** 2).mean(dim=-1, keepdim=True)
        return torch.cat((winner, neighbor, error), dim=-1)


def build() -> nn.Module:
    """Build a compact growing neural gas module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return GrowingNeuralGas()


def example_input() -> Tensor:
    """Return vector samples for gas adaptation.

    Returns
    -------
    Tensor
        Example tensor of shape ``(5, 4)``.
    """
    return torch.randn(5, 4)
