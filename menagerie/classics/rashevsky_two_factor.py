"""Rashevsky two-factor neuron, 1938, as a two-compartment dynamical unit.

Paper: Rashevsky 1938, "Mathematical Biophysics."
Excitatory and inhibitory factors integrate input with different time constants;
their difference crosses a threshold after accommodation-like dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Rashevsky two-factor neuron", "build", "example_input", "1938", "CA")]


class TwoFactorNeuron(nn.Module):
    """Euler-integrated excitatory and inhibitory neuron factors."""

    def __init__(self, n_units: int = 6, steps: int = 5, dt: float = 0.2) -> None:
        """Initialize factor gains and integration constants.

        Parameters
        ----------
        n_units
            Number of independent neuron channels.
        steps
            Number of Euler updates.
        dt
            Integration step size.
        """
        super().__init__()
        self.register_buffer("exc_gain", torch.linspace(0.8, 1.2, n_units))
        self.register_buffer("inh_gain", torch.linspace(0.25, 0.55, n_units))
        self.register_buffer("threshold", torch.linspace(0.15, 0.35, n_units))
        self.steps = steps
        self.dt = dt

    def forward(self, current: Tensor) -> Tensor:
        """Integrate two-factor dynamics for a fixed number of steps.

        Parameters
        ----------
        current
            Input current with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Thresholded firing probabilities.
        """
        exc = torch.zeros_like(current)
        inh = torch.zeros_like(current)
        for _ in range(self.steps):
            exc = exc + self.dt * (-exc / 1.0 + self.exc_gain * current)
            inh = inh + self.dt * (-inh / 2.5 + self.inh_gain * current)
        return torch.sigmoid(12.0 * (exc - inh - self.threshold))


def build() -> nn.Module:
    """Build a small Rashevsky two-factor module.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return TwoFactorNeuron()


def example_input() -> Tensor:
    """Create an input-current example.

    Returns
    -------
    Tensor
        Example current with shape ``(2, 6)``.
    """
    return torch.rand(2, 6)
