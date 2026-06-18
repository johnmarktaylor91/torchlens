"""Zhaoping Li V1 saliency network, 1999, Li Zhaoping.

Paper: "Contextual influences in V1 as a basis for pop out and asymmetry in visual
search." Orientation feature maps interact through recurrent lateral excitation and
inhibition to produce a saliency map.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Zhaoping Li V1 Saliency Network", "build", "example_input", "1999", "DA")]


class V1SaliencyNet(nn.Module):
    """Orientation-filter front end with recurrent lateral salience dynamics."""

    def __init__(self, n_orientations: int = 8, steps: int = 4) -> None:
        """Initialize feedforward and lateral orientation interactions.

        Parameters
        ----------
        n_orientations
            Number of orientation channels.
        steps
            Number of recurrent interaction steps.
        """
        super().__init__()
        self.front_end = nn.Conv2d(1, n_orientations, 9, padding=4, bias=False)
        self.lateral = nn.Conv2d(n_orientations, n_orientations, 5, padding=2, bias=False)
        self.steps = steps

    def forward(self, x: Tensor) -> Tensor:
        """Compute recurrent V1 saliency.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 128, 128)``.

        Returns
        -------
        Tensor
            Spatial saliency map.
        """
        feedforward = torch.relu(self.front_end(x))
        state = feedforward
        for _ in range(self.steps):
            state = torch.relu(feedforward + 0.25 * self.lateral(state))
        return state.max(dim=1, keepdim=True).values


def build() -> nn.Module:
    """Build a small V1 saliency module.

    Returns
    -------
    nn.Module
        Configured ``V1SaliencyNet`` instance.
    """
    return V1SaliencyNet()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 128, 128)``.
    """
    return torch.randn(1, 1, 128, 128)
