"""Mirollo-Strogatz pulse-coupled oscillators, 1990.

Paper: Mirollo and Strogatz 1990, "Synchronization of pulse-coupled biological
oscillators." Identical integrate-and-fire oscillators advance together through
global pulse coupling; this traceable version uses smooth firing rates instead
of discontinuous phase resets.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Mirollo-Strogatz pulse-coupled oscillators", "build", "example_input", "1990", "CF")
]


class MirolloStrogatz(nn.Module):
    """Smooth pulse-coupled phase oscillator population."""

    def __init__(self, epsilon: float = 0.08, dt: float = 0.12) -> None:
        """Initialize coupling and phase increment.

        Parameters
        ----------
        epsilon
            Global pulse coupling strength.
        dt
            Phase advance per step.
        """
        super().__init__()
        self.epsilon = epsilon
        self.dt = dt

    def forward(self, phases: Tensor) -> Tensor:
        """Roll out coupled oscillator phases.

        Parameters
        ----------
        phases
            Initial/reference phases of shape ``(batch, time, n_oscillators)``.

        Returns
        -------
        Tensor
            Concatenated smooth firing rates and order parameter per step.
        """
        phase = phases[:, 0].remainder(1.0)
        outputs: list[Tensor] = []
        for step in range(phases.shape[1]):
            drive = phases[:, step] * 0.02
            phase = phase + self.dt + drive
            fired = torch.sigmoid(30.0 * (phase - 1.0))
            phase = phase + self.epsilon * fired.sum(dim=-1, keepdim=True)
            phase = phase * (1.0 - fired)
            angle = 2.0 * torch.pi * phase
            order = torch.sqrt(
                torch.sin(angle).mean(dim=-1) ** 2 + torch.cos(angle).mean(dim=-1) ** 2
            )
            outputs.append(torch.cat((fired, order.unsqueeze(-1)), dim=-1))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a Mirollo-Strogatz oscillator module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return MirolloStrogatz()


def example_input() -> Tensor:
    """Return phase-driving inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 8, 5)``.
    """
    return torch.rand(2, 8, 5)
