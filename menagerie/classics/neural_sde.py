"""Tiny Neural SDE with learned drift and diffusion networks.

Paper: Kidger et al. 2020, "Neural SDEs as Infinite-Dimensional GANs."
"""

from __future__ import annotations

import torch
from torch import nn


class TinyNeuralSDE(nn.Module):
    """Diagonal Ito Neural SDE integrated with a few Euler-Maruyama steps."""

    def __init__(self, state_dim: int = 4, hidden_dim: int = 8, steps: int = 2) -> None:
        """Initialize drift and diffusion networks.

        Parameters
        ----------
        state_dim:
            Latent state width.
        hidden_dim:
            Hidden width for drift and diffusion MLPs.
        steps:
            Number of fixed integration steps.
        """

        super().__init__()
        self.steps = steps
        self.drift = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        noise = torch.tensor([[0.30, -0.20, 0.10, -0.15], [-0.05, 0.25, -0.30, 0.20]])
        self.register_buffer("noise", noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate the SDE from ``t=0`` to ``t=1``.

        Parameters
        ----------
        x:
            Initial state tensor of shape ``(batch, state_dim)``.

        Returns
        -------
        torch.Tensor
            Final latent state.
        """

        y = x
        dt = 1.0 / float(self.steps)
        sqrt_dt = dt**0.5
        for step in range(self.steps):
            t = y.new_full((y.shape[0], 1), float(step) * dt)
            state_time = torch.cat([y, t], dim=-1)
            drift = self.drift(state_time)
            diffusion = torch.tanh(self.diffusion(state_time))
            d_w = self.noise[step].unsqueeze(0).expand_as(y) * sqrt_dt
            y = y + drift * dt + diffusion * d_w
        return y


def build() -> nn.Module:
    """Build a tiny Neural SDE.

    Returns
    -------
    nn.Module
        Random-initialized Neural SDE.
    """

    return TinyNeuralSDE().eval()


def example_input() -> torch.Tensor:
    """Create a small initial latent state.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 4)``.
    """

    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [("neural_sde", "build", "example_input", "2020", "E6")]
