"""FINN: Finite Volume Neural Network.

Paper: "Finite Volume Neural Network: Modeling Subsurface Contaminant
Transport", Praditia et al., 2021.

The reconstruction keeps FINN's finite-volume skeleton: learnable constitutive
flux kernels compute interface fluxes, boundary states are explicit, and a
state kernel advances a conservation-law update over control volumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FluxKernel(nn.Module):
    """Learnable interface flux function."""

    def __init__(self, hidden: int = 16) -> None:
        """Initialize a neural flux kernel."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute flux between neighboring volumes."""

        return self.net(torch.stack([left, right], dim=-1)).squeeze(-1)


class FINNCompact(nn.Module):
    """Compact 1-D FINN update module."""

    def __init__(self, steps: int = 4, dx: float = 1.0, dt: float = 0.1) -> None:
        """Initialize finite-volume dynamics."""

        super().__init__()
        self.steps = steps
        self.dx = dx
        self.dt = dt
        self.flux = FluxKernel()
        self.state_kernel = nn.Sequential(
            nn.Linear(1, 8), nn.Sigmoid(), nn.Linear(8, 1), nn.Softplus()
        )
        self.left_boundary = nn.Parameter(torch.tensor(0.0))
        self.right_boundary = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Roll out conservative finite-volume updates."""

        u = state.squeeze(1)
        states = []
        for _ in range(self.steps):
            left = torch.cat([self.left_boundary.expand(u.shape[0], 1), u[:, :-1]], dim=1)
            right = torch.cat([u[:, 1:], self.right_boundary.expand(u.shape[0], 1)], dim=1)
            flux_left = self.flux(left, u)
            flux_right = self.flux(u, right)
            retardation = self.state_kernel(u.unsqueeze(-1)).squeeze(-1) + 1.0
            u = u - (self.dt / self.dx) * (flux_right - flux_left) / retardation
            states.append(u.unsqueeze(1))
        return torch.stack(states, dim=1)


def build() -> nn.Module:
    """Build compact FINN."""

    return FINNCompact()


def example_input() -> torch.Tensor:
    """Return a 1-D concentration field."""

    return torch.randn(1, 1, 24)


MENAGERIE_ENTRIES = [("FINN", "build", "example_input", "2021", "E7")]
