"""Contrastive Hebbian Learning network, 1986.

Movellan and related CHL equilibrium nets settle symmetric connections in free and
clamped phases; the traceable architecture is the recurrent settling computation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContrastiveHebbianNet(nn.Module):
    """Symmetric recurrent settling network used by CHL."""

    def __init__(self, n_in: int = 8, n_hidden: int = 6, n_out: int = 4, steps: int = 6) -> None:
        """Initialize layered symmetric weights.

        Parameters
        ----------
        n_in
            Number of input units.
        n_hidden
            Number of hidden units.
        n_out
            Number of output units.
        steps
            Number of settling iterations.
        """
        super().__init__()
        self.steps = steps
        self.w_xh = nn.Parameter(torch.randn(n_in, n_hidden) * 0.2)
        self.w_hy = nn.Parameter(torch.randn(n_hidden, n_out) * 0.2)
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.output_bias = nn.Parameter(torch.zeros(n_out))

    def settle(self, x: Tensor, clamped_y: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Settle hidden and output rates with optional output clamp.

        Parameters
        ----------
        x
            Input pattern.
        clamped_y
            Optional target output rates for the plus phase.

        Returns
        -------
        tuple[Tensor, Tensor]
            Hidden and output rates.
        """
        h = torch.sigmoid(x @ self.w_xh + self.hidden_bias)
        y = torch.sigmoid(h @ self.w_hy + self.output_bias)
        for _ in range(self.steps):
            h = torch.sigmoid(x @ self.w_xh + y @ self.w_hy.t() + self.hidden_bias)
            free_y = torch.sigmoid(h @ self.w_hy + self.output_bias)
            y = free_y if clamped_y is None else 0.5 * free_y + 0.5 * clamped_y
        return h, y

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run the free/minus phase used by CHL.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output and hidden rates after settling.
        """
        h, y = self.settle(x)
        return y, h


def build() -> nn.Module:
    """Build a small random contrastive-Hebbian network.

    Returns
    -------
    nn.Module
        Random-initialized CHL settling module.
    """
    return ContrastiveHebbianNet()


def example_input() -> Tensor:
    """Return a float32 example input.

    Returns
    -------
    Tensor
        Example input of shape ``(2, 8)``.
    """
    return torch.randn(2, 8, dtype=torch.float32)
