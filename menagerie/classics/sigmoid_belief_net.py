"""Sigmoid belief network, 1992, Radford Neal.

Paper: Neal 1992, "Connectionist learning of belief networks."
Directed stochastic binary layers are represented by differentiable mean-field sigmoid
conditionals; ancestral sampling and MCMC training are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SigmoidBeliefNetwork(nn.Module):
    """Layered directed belief net using mean-field sigmoid probabilities."""

    def __init__(self, visible_dim: int = 64, hidden_dim: int = 24, top_dim: int = 12) -> None:
        """Initialize bottom-up recognition and top-down generative maps.

        Parameters
        ----------
        visible_dim
            Number of visible features.
        hidden_dim
            Hidden layer width.
        top_dim
            Top latent width.
        """
        super().__init__()
        self.recognition = nn.Linear(visible_dim, top_dim)
        self.top_to_hidden = nn.Linear(top_dim, hidden_dim)
        self.hidden_to_visible = nn.Linear(hidden_dim, visible_dim)

    def forward(self, visible: Tensor) -> Tensor:
        """Compute top-down visible probabilities from mean-field latents.

        Parameters
        ----------
        visible
            Visible feature tensor with shape ``(batch, 64)``.

        Returns
        -------
        Tensor
            Reconstructed visible probabilities.
        """
        top = torch.sigmoid(self.recognition(visible))
        hidden = torch.sigmoid(self.top_to_hidden(top))
        return torch.sigmoid(self.hidden_to_visible(hidden))


MENAGERIE_ENTRIES = [("Sigmoid Belief Network", "build", "example_input", "1992", "DA")]


def build() -> nn.Module:
    """Build a sigmoid belief network.

    Returns
    -------
    nn.Module
        Configured SBN module.
    """
    return SigmoidBeliefNetwork()


def example_input() -> Tensor:
    """Create visible probabilities.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.rand(1, 64)
