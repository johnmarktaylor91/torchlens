"""Deep Tensor Neural Network, 2017, Schutt, Mueller, and Tkatchenko.

Paper: Schutt 2017, "Quantum-chemical insights from deep tensor neural networks."
Atom embeddings interact through distance-conditioned tensor filters and are summed into
a molecular observable; this standalone version omits dataset featurization and training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DTNN(nn.Module):
    """Distance-conditioned atom interaction network."""

    def __init__(self, n_atom_types: int = 10, hidden_size: int = 24, passes: int = 3) -> None:
        """Initialize embeddings, distance filters, and energy readout.

        Parameters
        ----------
        n_atom_types
            Number of coarse atom-type bins.
        hidden_size
            Atom embedding width.
        passes
            Number of interaction passes.
        """
        super().__init__()
        self.passes = passes
        self.embedding = nn.Embedding(n_atom_types, hidden_size)
        self.distance_filter = nn.Linear(1, hidden_size)
        self.atom_filter = nn.Linear(hidden_size, hidden_size)
        self.update = nn.Linear(hidden_size, hidden_size)
        self.energy = nn.Linear(hidden_size, 1)

    def forward(self, packed: Tensor) -> Tensor:
        """Compute a molecular observable from packed coordinates and atom features.

        Parameters
        ----------
        packed
            Tensor with shape ``(batch, 12, 4)``; first three channels are coordinates and
            the last channel is a coarse atom-type value.

        Returns
        -------
        Tensor
            Molecular energy-like scalar.
        """
        coords = packed[:, :, :3]
        atom_values = packed[:, :, 3].abs().mul(10.0).long().remainder(10)
        state = self.embedding(atom_values)
        distances = torch.cdist(coords, coords).unsqueeze(-1)
        for _ in range(self.passes):
            d_filter = torch.tanh(self.distance_filter(distances))
            neighbor = self.atom_filter(state).unsqueeze(1)
            messages = torch.sum(d_filter * neighbor, dim=2)
            state = state + torch.tanh(self.update(messages))
        return self.energy(state).sum(dim=1)


MENAGERIE_ENTRIES = [("DTNN (Deep Tensor Neural Network)", "build", "example_input", "2017", "DA")]


def build() -> nn.Module:
    """Build a compact DTNN.

    Returns
    -------
    nn.Module
        Configured DTNN module.
    """
    return DTNN()


def example_input() -> Tensor:
    """Create packed coordinates and coarse atom values.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 12, 4)``.
    """
    coords = torch.randn(1, 12, 3)
    atom_values = torch.randint(0, 10, (1, 12, 1), dtype=torch.float32) / 10.0
    return torch.cat((coords, atom_values), dim=-1)
