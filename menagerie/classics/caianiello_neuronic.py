"""Caianiello neuronic equations, 1961, as delayed threshold dynamics.

Paper: Caianiello 1961, "Outline of a Theory of Thought-Processes and Thinking Machines."
Discrete threshold neurons integrate weighted delayed states through a temporal
memory kernel, represented here by a compact recurrent history buffer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Caianiello neuronic equations", "build", "example_input", "1961", "CA")]


class NeuronicNet(nn.Module):
    """Delayed threshold network with straight-through Heaviside outputs."""

    def __init__(self, n_units: int = 6, delays: int = 3, steps: int = 4) -> None:
        """Initialize delay-tap weights and thresholds.

        Parameters
        ----------
        n_units
            Number of threshold units.
        delays
            Number of delayed history taps.
        steps
            Number of recurrent ticks.
        """
        super().__init__()
        weights = torch.randn(n_units, n_units, delays) * 0.35
        weights = weights + torch.eye(n_units).unsqueeze(-1) * torch.tensor([0.7, 0.3, 0.1])
        self.weight = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.2, n_units))
        self.steps = steps
        self.delays = delays

    def _heaviside_ste(self, drive: Tensor) -> Tensor:
        """Apply a traceable straight-through threshold.

        Parameters
        ----------
        drive
            Pre-threshold activations.

        Returns
        -------
        Tensor
            Binary-like activations with sigmoid surrogate gradients.
        """
        soft = torch.sigmoid(12.0 * drive)
        hard = (drive >= 0.0).to(drive.dtype)
        return hard.detach() - soft.detach() + soft

    def forward(self, state: Tensor) -> Tensor:
        """Run delayed neuronic dynamics from an initial state.

        Parameters
        ----------
        state
            Initial unit state with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Final binary-like unit state.
        """
        history = state.unsqueeze(-1).repeat(1, 1, self.delays)
        current = state
        for _ in range(self.steps):
            drive = torch.einsum("bik,jik->bj", history, self.weight) + self.bias
            current = self._heaviside_ste(drive)
            history = torch.cat((current.unsqueeze(-1), history[:, :, :-1]), dim=-1)
        return current


def build() -> nn.Module:
    """Build a small Caianiello neuronic network.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return NeuronicNet()


def example_input() -> Tensor:
    """Create an initial binary state.

    Returns
    -------
    Tensor
        Example state with shape ``(2, 6)``.
    """
    return torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
