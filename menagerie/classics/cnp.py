"""Conditional Neural Process (CNP), compact faithful reconstruction.

Garnelo et al. (2018) proposed CNPs as neural-process models that condition on
sets of observed context pairs and predict target values by encoding each
``(x_i, y_i)`` pair, aggregating the representations with a permutation-invariant
mean, and decoding each target coordinate conditioned on that aggregate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionalNeuralProcess(nn.Module):
    """Set-conditioned encoder-aggregator-decoder CNP."""

    def __init__(self, x_dim: int = 1, y_dim: int = 1, hidden: int = 64) -> None:
        """Initialize CNP encoder and decoder.

        Parameters
        ----------
        x_dim:
            Coordinate dimension.
        y_dim:
            Observation dimension.
        hidden:
            Representation width.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden + x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, y_dim * 2),
        )
        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(self, packed: Tensor) -> Tensor:
        """Predict target distributions from packed context and target coordinates.

        Parameters
        ----------
        packed:
            Tensor of shape ``(batch, context + target, x_dim + y_dim)``. Context
            rows contain ``x,y`` pairs; target rows use the first ``x_dim`` values
            as coordinates and ignore the remaining slots.

        Returns
        -------
        Tensor
            Mean and positive scale for each target point.
        """
        context = packed[:, :8]
        target_x = packed[:, 8:, : self.x_dim]
        encoded = self.encoder(context)
        representation = encoded.mean(dim=1, keepdim=True)
        representation = representation.expand(-1, target_x.shape[1], -1)
        stats = self.decoder(torch.cat((target_x, representation), dim=-1))
        mean, raw_scale = stats.chunk(2, dim=-1)
        return torch.cat((mean, torch.nn.functional.softplus(raw_scale) + 1e-3), dim=-1)


def build() -> nn.Module:
    """Build a compact CNP.

    Returns
    -------
    nn.Module
        Random-init Conditional Neural Process.
    """
    return ConditionalNeuralProcess()


def example_input() -> Tensor:
    """Return packed context and target rows.

    Returns
    -------
    Tensor
        Packed CNP input.
    """
    x_context = torch.linspace(-1.0, 1.0, 8).view(1, 8, 1)
    y_context = torch.sin(x_context * 3.14159)
    x_target = torch.linspace(-1.25, 1.25, 12).view(1, 12, 1)
    y_blank = torch.zeros_like(x_target)
    return torch.cat((torch.cat((x_context, y_context), -1), torch.cat((x_target, y_blank), -1)), 1)


MENAGERIE_ENTRIES = [("CNP", "build", "example_input", "2018", "neural-process")]
