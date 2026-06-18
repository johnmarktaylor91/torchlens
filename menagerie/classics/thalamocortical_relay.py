"""Thalamocortical relay-gating loop, 1998, thalamocortical circuit models.

Paper: "The thalamus as a monitor of motor outputs." A relay transform is
multiplicatively gated by corticothalamic feedback, then fed through a small
cortical recurrence to model tonic/burst-like gain control.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Thalamocortical Relay-Gating Loop", "build", "example_input", "1998", "DB")]


class ThalamocorticalRelay(nn.Module):
    """Gated thalamic relay with cortical recurrent loop."""

    def __init__(self, n_input: int = 64, n_hidden: int = 32, steps: int = 3) -> None:
        """Initialize relay, feedback gate, and cortical transform.

        Parameters
        ----------
        n_input
            Input and feedback dimensionality.
        n_hidden
            Relay hidden dimensionality.
        steps
            Number of loop iterations.
        """
        super().__init__()
        self.relay = nn.Linear(n_input, n_hidden)
        self.gate = nn.Linear(n_input, n_hidden)
        self.cortex = nn.Linear(n_hidden, n_input)
        self.steps = steps

    def forward(self, x: Tensor) -> Tensor:
        """Run a recurrent thalamocortical gating loop.

        Parameters
        ----------
        x
            Sensory drive tensor of shape ``(batch, n_input)``.

        Returns
        -------
        Tensor
            Cortical output after relay gating.
        """
        feedback = x
        cortical = x
        for _ in range(self.steps):
            relay = self.relay(x) * torch.sigmoid(self.gate(feedback))
            cortical = torch.tanh(self.cortex(relay))
            feedback = cortical
        return cortical


def build() -> nn.Module:
    """Build a small thalamocortical relay module.

    Returns
    -------
    nn.Module
        Configured ``ThalamocorticalRelay`` instance.
    """
    return ThalamocorticalRelay()


def example_input() -> Tensor:
    """Return a sensory drive example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)
