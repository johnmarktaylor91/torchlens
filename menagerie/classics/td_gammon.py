"""TD-Gammon value network, 1992, Tesauro.

Paper: "Practical issues in temporal difference learning."
The network is the compact feedforward value substrate used for TD(lambda) self-play;
the temporal-difference update and game engine are intentionally outside this module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TDGammonValueNetwork(nn.Module):
    """Two-layer sigmoid MLP mapping board features to outcome probabilities."""

    def __init__(self, n_features: int = 198, hidden_size: int = 40, n_out: int = 4) -> None:
        """Initialize the TD-Gammon value network.

        Parameters
        ----------
        n_features
            Number of encoded backgammon board features.
        hidden_size
            Number of hidden units.
        n_out
            Number of outcome probability channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, n_out),
            nn.Sigmoid(),
        )

    def forward(self, board_encoding: Tensor) -> Tensor:
        """Estimate outcome probabilities from board features.

        Parameters
        ----------
        board_encoding
            Encoded board tensor of shape ``(batch, 198)``.

        Returns
        -------
        Tensor
            Outcome probabilities.
        """
        return self.net(board_encoding)


def build() -> nn.Module:
    """Build a small TD-Gammon value network.

    Returns
    -------
    nn.Module
        Configured ``TDGammonValueNetwork`` instance.
    """
    return TDGammonValueNetwork()


def example_input() -> Tensor:
    """Create an example board encoding.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 198)``.
    """
    return torch.randn(1, 198)


MENAGERIE_ENTRIES = [("TD-Gammon Value Network", "build", "example_input", "1992", "DD")]
