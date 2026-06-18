"""Rao-Ballard hierarchical predictive coding, 1999.

Rao and Ballard, "Predictive coding in the visual cortex: a functional interpretation
of some extra-classical receptive-field effects." A top-down hierarchy predicts lower
levels while bottom-up residuals iteratively refine latent causes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RaoBallardPCN(nn.Module):
    """Small Rao-Ballard predictive-coding hierarchy."""

    def __init__(self, n_in: int = 16, n_hidden: int = 8, steps: int = 4, dt: float = 0.2) -> None:
        """Initialize random top-down dictionaries.

        Parameters
        ----------
        n_in
            Number of input units.
        n_hidden
            Number of hidden cause units.
        steps
            Number of inference updates to unroll.
        dt
            Euler update size for latent causes.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.bottom_decoder = nn.Linear(n_hidden, n_in, bias=False)
        self.top_decoder = nn.Linear(n_hidden, n_hidden, bias=False)
        self.lateral = nn.Linear(n_hidden, n_hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Infer hidden causes and return reconstruction, residuals, and state.

        Parameters
        ----------
        x
            Input activity of shape ``(batch, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, input residual, and final hidden representation.
        """
        r = torch.tanh(self.lateral(x.new_zeros(x.shape[0], self.top_decoder.in_features)))
        for _ in range(self.steps):
            pred_x = self.bottom_decoder(r)
            err_x = x - pred_x
            pred_r = torch.tanh(self.top_decoder(r))
            err_r = r - pred_r
            grad = F.linear(err_x, self.bottom_decoder.weight.t()) - err_r + self.lateral(r)
            r = torch.tanh(r + self.dt * grad)
        reconstruction = self.bottom_decoder(r)
        residual = x - reconstruction
        return reconstruction, residual, r


def build() -> nn.Module:
    """Build a small random Rao-Ballard predictive-coding module.

    Returns
    -------
    nn.Module
        Random-initialized predictive-coding network.
    """
    return RaoBallardPCN()


def example_input() -> Tensor:
    """Return a float32 example input.

    Returns
    -------
    Tensor
        Example input of shape ``(2, 16)``.
    """
    return torch.randn(2, 16, dtype=torch.float32)
