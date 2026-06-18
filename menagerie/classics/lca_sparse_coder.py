"""Locally Competitive Algorithm sparse coding, 2008, Rozell and colleagues.

Paper: "Sparse coding via thresholding and local competition in neural circuits."
Recurrent membrane potentials integrate feedforward drive and lateral inhibition,
then soft-threshold into sparse coefficients for reconstruction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Locally Competitive Algorithm (LCA) Sparse Coding", "build", "example_input", "2008", "DA")
]


class LCASparseCoder(nn.Module):
    """Recurrent sparse coder with lateral dictionary competition."""

    def __init__(self, n_input: int = 128, n_atoms: int = 48, steps: int = 5) -> None:
        """Initialize dictionary and LCA integration settings.

        Parameters
        ----------
        n_input
            Input dimensionality.
        n_atoms
            Number of dictionary atoms.
        steps
            Number of recurrent inference steps.
        """
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn(n_atoms, n_input) * 0.08)
        self.steps = steps
        self.dt = 0.2
        self.lam = 0.1

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Infer sparse coefficients and reconstruct inputs.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Sparse coefficients and linear reconstruction.
        """
        gram = self.dictionary @ self.dictionary.T
        lateral = gram - torch.eye(gram.shape[0], device=x.device, dtype=x.dtype)
        drive = x @ self.dictionary.T
        u = torch.zeros_like(drive)
        a = torch.zeros_like(drive)
        for _ in range(self.steps):
            du = drive - u - a @ lateral
            u = u + self.dt * du
            a = torch.sign(u) * F.relu(torch.abs(u) - self.lam)
        return a, a @ self.dictionary


def build() -> nn.Module:
    """Build a small LCA sparse coder.

    Returns
    -------
    nn.Module
        Configured ``LCASparseCoder`` instance.
    """
    return LCASparseCoder()


def example_input() -> Tensor:
    """Return a vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 128)``.
    """
    return torch.randn(1, 128)
