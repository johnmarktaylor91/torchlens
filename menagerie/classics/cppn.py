"""CPPN coordinate pattern-producing network, 2007, Stanley.

Paper: "Compositional pattern producing networks: A novel abstraction of development."
Coordinates pass through heterogeneous activation channels to encode regular spatial
patterns; evolution of the graph is omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CPPNCoordinatePatternNetwork(nn.Module):
    """Small feedforward CPPN with mixed per-channel activations."""

    def __init__(self, n_in: int = 4, hidden_size: int = 16, n_out: int = 3) -> None:
        """Initialize coordinate transforms.

        Parameters
        ----------
        n_in
            Number of coordinate features, typically ``x, y, r, bias``.
        hidden_size
            Number of heterogeneous hidden channels.
        n_out
            Number of output pattern channels.
        """
        super().__init__()
        self.hidden = nn.Linear(n_in, hidden_size)
        self.out = nn.Linear(hidden_size, n_out)

    def _mixed_activation(self, z: Tensor) -> Tensor:
        """Apply round-robin CPPN activation families.

        Parameters
        ----------
        z
            Hidden preactivation tensor.

        Returns
        -------
        Tensor
            Activated hidden tensor.
        """
        chunks = torch.chunk(z, 4, dim=-1)
        activated = [
            torch.sin(chunks[0]),
            torch.exp(-(chunks[1] * chunks[1])),
            torch.tanh(chunks[2]),
            torch.abs(chunks[3]),
        ]
        return torch.cat(activated, dim=-1)

    def forward(self, coords: Tensor) -> Tensor:
        """Generate a spatial pattern from coordinates.

        Parameters
        ----------
        coords
            Coordinate tensor of shape ``(batch, 4)``.

        Returns
        -------
        Tensor
            Pattern channels.
        """
        return torch.tanh(self.out(self._mixed_activation(self.hidden(coords))))


def build() -> nn.Module:
    """Build a small CPPN.

    Returns
    -------
    nn.Module
        Configured ``CPPNCoordinatePatternNetwork`` instance.
    """
    return CPPNCoordinatePatternNetwork()


def example_input() -> Tensor:
    """Create an example coordinate vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.tensor([[0.25, -0.5, 0.5590, 1.0]])


MENAGERIE_ENTRIES = [
    ("CPPN Coordinate Pattern-Producing Network", "build", "example_input", "2007", "DD")
]
