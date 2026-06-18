"""Friston free-energy predictive-coding net, 2005.

Friston, "A theory of cortical responses." Hidden causes are iteratively updated to
minimize precision-weighted prediction errors, yielding a variational free-energy scalar.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FristonFreeEnergyNet(nn.Module):
    """Precision-weighted predictive-coding network."""

    def __init__(self, n_in: int = 10, n_cause: int = 5, steps: int = 5, dt: float = 0.2) -> None:
        """Initialize generative weights and log-precisions.

        Parameters
        ----------
        n_in
            Number of observed units.
        n_cause
            Number of hidden causes.
        steps
            Number of variational inference updates.
        dt
            Euler update size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.generative = nn.Linear(n_cause, n_in, bias=False)
        self.prior = nn.Linear(n_cause, n_cause)
        self.log_precision_x = nn.Parameter(torch.zeros(n_in))
        self.log_precision_z = nn.Parameter(torch.zeros(n_cause))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Infer causes and report prediction errors plus free energy.

        Parameters
        ----------
        x
            Observation tensor of shape ``(batch, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Prediction, sensory error, hidden cause, and per-example free energy.
        """
        z = x.new_zeros(x.shape[0], self.generative.in_features)
        precision_x = torch.exp(self.log_precision_x)
        precision_z = torch.exp(self.log_precision_z)
        for _ in range(self.steps):
            prediction = self.generative(z)
            err_x = (x - prediction) * precision_x
            prior_pred = torch.tanh(self.prior(z))
            err_z = (z - prior_pred) * precision_z
            grad = torch.matmul(err_x, self.generative.weight) - err_z
            z = z + self.dt * grad
        final_prediction = self.generative(z)
        final_err_x = x - final_prediction
        free_energy = 0.5 * (
            (final_err_x.square() * precision_x).sum(dim=-1)
            + ((z - torch.tanh(self.prior(z))).square() * precision_z).sum(dim=-1)
        )
        return final_prediction, final_err_x, z, free_energy


def build() -> nn.Module:
    """Build a small random Friston free-energy network.

    Returns
    -------
    nn.Module
        Random-initialized free-energy predictive-coding network.
    """
    return FristonFreeEnergyNet()


def example_input() -> Tensor:
    """Return a float32 example observation.

    Returns
    -------
    Tensor
        Example input of shape ``(2, 10)``.
    """
    return torch.randn(2, 10, dtype=torch.float32)
