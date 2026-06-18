"""Continuous Attractor Neural Network, 1995.

Ring-attractor models for head direction and working memory use translation-invariant
Mexican-hat recurrent weights to sustain a continuum of stable activity bumps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContinuousAttractorNet(nn.Module):
    """Hand-wired ring attractor with cosine recurrent weights."""

    def __init__(self, n_units: int = 32, steps: int = 8, dt: float = 0.15) -> None:
        """Initialize ring weights and preferred angles.

        Parameters
        ----------
        n_units
            Number of ring units.
        steps
            Number of recurrent updates.
        dt
            Euler update size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        theta = torch.linspace(0.0, 2.0 * torch.pi, n_units + 1)[:-1]
        diff = theta[:, None] - theta[None, :]
        weights = -0.1 / n_units + 1.2 * torch.cos(diff) / n_units
        self.register_buffer("theta", theta)
        self.register_buffer("weights", weights)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Settle a ring bump under external input.

        Parameters
        ----------
        x
            External input of shape ``(batch, n_units)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Final rates and decoded bump angle vector ``(cos, sin)``.
        """
        r = torch.relu(x)
        for _ in range(self.steps):
            drive = torch.matmul(r, self.weights.t()) + x
            r = r + self.dt * (-r + torch.relu(drive))
            r = r / (r.sum(dim=-1, keepdim=True).clamp_min(1e-6))
        decoded = torch.stack(
            ((r * torch.cos(self.theta)).sum(dim=-1), (r * torch.sin(self.theta)).sum(dim=-1)),
            dim=-1,
        )
        return r, decoded


def build() -> nn.Module:
    """Build a ring-attractor CANN.

    Returns
    -------
    nn.Module
        Continuous attractor module.
    """
    return ContinuousAttractorNet()


def example_input() -> Tensor:
    """Return a float32 ring input.

    Returns
    -------
    Tensor
        Input of shape ``(2, 32)``.
    """
    x = torch.zeros(2, 32, dtype=torch.float32)
    x[:, 4:8] = 1.0
    return x
