"""Mumford predictive cortical architecture, 1992.

Mumford, "On the computational architecture of the neocortex." A small
analysis-by-synthesis loop combines top-down hypotheses with bottom-up residual support.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MumfordCortical(nn.Module):
    """Recurrent hypothesis/error hierarchy in Mumford's cortical style."""

    def __init__(self, n_in: int = 12, n_state: int = 6, steps: int = 5, dt: float = 0.25) -> None:
        """Initialize the paired representation and error populations.

        Parameters
        ----------
        n_in
            Number of sensory input units.
        n_state
            Number of hypothesis units.
        steps
            Number of recurrent analysis-by-synthesis iterations.
        dt
            Euler update size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.decoder = nn.Linear(n_state, n_in, bias=False)
        self.support = nn.Linear(n_in, n_state, bias=False)
        self.lateral = nn.Linear(n_state, n_state)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run recurrent top-down prediction and bottom-up support.

        Parameters
        ----------
        x
            Sensory input of shape ``(batch, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Final prediction, residual population, and hypothesis state.
        """
        h = torch.tanh(self.support(x))
        for _ in range(self.steps):
            prediction = self.decoder(h)
            residual = x - prediction
            normalized_support = F.normalize(self.support(residual), dim=-1)
            recurrent_context = torch.tanh(self.lateral(h))
            h = torch.tanh(h + self.dt * (normalized_support + recurrent_context - h))
        final_prediction = self.decoder(h)
        final_residual = x - final_prediction
        return final_prediction, final_residual, h


def build() -> nn.Module:
    """Build a small random Mumford predictive-cortical module.

    Returns
    -------
    nn.Module
        Random-initialized predictive-cortical network.
    """
    return MumfordCortical()


def example_input() -> Tensor:
    """Return a float32 example input.

    Returns
    -------
    Tensor
        Example input of shape ``(2, 12)``.
    """
    return torch.randn(2, 12, dtype=torch.float32)
