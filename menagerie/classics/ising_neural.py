"""Cragg-Temperley Ising-analogy neural net, 1954, with Glauber-like updates.

Paper: Cragg and Temperley 1954, "The Organization of Neurones: A Cooperative Analogy."
Ising-spin neurons flip according to temperature-scaled energy differences,
prefiguring later Hopfield and Boltzmann-style stochastic neural networks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Cragg-Temperley Ising-analogy neural net", "build", "example_input", "1954", "CA")
]


class IsingNeuralNet(nn.Module):
    """Small synchronous Ising neural net with soft Glauber flip probabilities."""

    def __init__(self, n_units: int = 8, sweeps: int = 4, beta: float = 2.0) -> None:
        """Initialize symmetric couplings and local fields.

        Parameters
        ----------
        n_units
            Number of spin neurons.
        sweeps
            Number of update sweeps.
        beta
            Inverse temperature.
        """
        super().__init__()
        raw = torch.randn(n_units, n_units) * 0.25
        coupling = (raw + raw.T) / 2
        coupling.fill_diagonal_(0.0)
        self.register_buffer("coupling", coupling)
        self.register_buffer("field", torch.linspace(-0.2, 0.2, n_units))
        self.sweeps = sweeps
        self.beta = beta

    def forward(self, spins: Tensor) -> Tensor:
        """Relax bipolar spins with differentiable Glauber-style flips.

        Parameters
        ----------
        spins
            Bipolar spin state of shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Relaxed bipolar-like spin state.
        """
        state = spins.clamp(-1.0, 1.0)
        for _ in range(self.sweeps):
            local = state @ self.coupling + self.field
            delta_energy = 2.0 * state * local
            flip_prob = torch.sigmoid(-self.beta * delta_energy)
            state = state * (1.0 - 2.0 * flip_prob)
        return state


def build() -> nn.Module:
    """Build a small Ising neural net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return IsingNeuralNet()


def example_input() -> Tensor:
    """Create a bipolar spin example.

    Returns
    -------
    Tensor
        Example spins with shape ``(2, 8)``.
    """
    return torch.sign(torch.randn(2, 8))
