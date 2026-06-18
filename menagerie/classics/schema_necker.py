"""McClelland-Rumelhart schema constraint-satisfaction net, 1986, for Necker-style relaxation.

Paper: McClelland and Rumelhart 1986, "Parallel Distributed Processing, Volume 2."
Symmetric micro-hypothesis weights relax toward a coherent schema, illustrated
with a compact Necker-cube-like constraint matrix and deterministic settling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    (
        "McClelland-Rumelhart schema constraint-satisfaction net (Necker)",
        "build",
        "example_input",
        "1986",
        "CB",
    )
]


class SchemaNecker(nn.Module):
    """Symmetric schema relaxation network."""

    def __init__(self, n_units: int = 16, steps: int = 8) -> None:
        """Initialize Necker-style compatible and incompatible constraints.

        Parameters
        ----------
        n_units
            Number of micro-hypothesis units.
        steps
            Number of relaxation steps.
        """
        super().__init__()
        weights = torch.zeros(n_units, n_units)
        half = n_units // 2
        weights[:half, :half] = 0.12
        weights[half:, half:] = 0.12
        weights[:half, half:] = -0.18
        weights[half:, :half] = -0.18
        weights.fill_diagonal_(0.0)
        self.register_buffer("weights", weights)
        self.register_buffer("bias", torch.linspace(-0.05, 0.05, n_units))
        self.steps = steps

    def forward(self, initial: Tensor) -> Tensor:
        """Relax schema activations from an initial or clamp vector.

        Parameters
        ----------
        initial
            Initial activation with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Settled schema activations in ``[0, 1]``.
        """
        activation = initial.clamp(0.0, 1.0)
        for _ in range(self.steps):
            drive = activation @ self.weights + self.bias + initial
            activation = torch.sigmoid(activation + 0.6 * drive)
        return activation


def build() -> nn.Module:
    """Build a small schema relaxation net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return SchemaNecker()


def example_input() -> Tensor:
    """Create a Necker-style ambiguous initial state.

    Returns
    -------
    Tensor
        Example state with shape ``(1, 16)``.
    """
    x = torch.full((1, 16), 0.25)
    x[:, [0, 3, 8, 11]] = 0.7
    return x
