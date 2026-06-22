"""DeepXDE FNN PINN compact physics-informed neural network.

Lu et al., "DeepXDE: A Deep Learning Library for Solving Differential
Equations", SIAM Review 2021.  The common DeepXDE PINN model is a fully
connected neural network whose coordinate-output map is differentiated to form
PDE residual losses.  TorchLens inference-only tracing rejects backward-engine
calls, so this compact classic returns the solution value and a traceable
finite-difference Poisson residual surrogate in the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepXDEFNNPINN(nn.Module):
    """Fully connected PINN with PDE residual output."""

    def __init__(self, width: int = 32) -> None:
        """Initialize tanh FNN layers.

        Parameters
        ----------
        width:
            Hidden layer width.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return solution values and a Poisson residual proxy.

        Parameters
        ----------
        x:
            Coordinate tensor ``(B, 2)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Network solution and finite-difference residual-like term.
        """

        eps = 1.0e-2
        u = self.net(x)
        lap_terms = []
        for dim in range(x.shape[-1]):
            offset = torch.zeros_like(x)
            offset[:, dim] = eps
            up = self.net(x + offset)
            um = self.net(x - offset)
            lap_terms.append((up - 2.0 * u + um) / (eps * eps))
        residual = sum(lap_terms) + torch.sin(x[:, :1]) * torch.cos(x[:, 1:2])
        return u, residual


def build() -> nn.Module:
    """Build a compact DeepXDE-style PINN.

    Returns
    -------
    nn.Module
        Fully connected PINN.
    """

    return DeepXDEFNNPINN()


def example_input() -> torch.Tensor:
    """Create coordinate collocation points.

    Returns
    -------
    torch.Tensor
        Example tensor ``(8, 2)``.
    """

    return torch.randn(8, 2)


MENAGERIE_ENTRIES = [
    (
        "DeepXDE FNN PINN (coordinate FNN with PDE-residual neural field)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
