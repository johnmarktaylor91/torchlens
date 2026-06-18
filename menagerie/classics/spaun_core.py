"""Spaun integrated brain model core substrate, 2012, Eliasmith and colleagues.

Paper: "A large-scale model of the functioning brain." This simplified substrate
keeps the forward pathway vision-to-working-memory-to-BG-action-to-motor while
omitting the full eight-task SPA/NEF implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Spaun Integrated Brain Model (core substrate)", "build", "example_input", "2012", "DB")
]


class SpaunCore(nn.Module):
    """Simplified vision, working-memory, action-selection, and motor pathway."""

    def __init__(self, n_input: int = 784, spa_dim: int = 64, n_actions: int = 8) -> None:
        """Initialize the core Spaun-inspired modules.

        Parameters
        ----------
        n_input
            Flattened visual input dimensionality.
        spa_dim
            Semantic-pointer working-memory dimensionality.
        n_actions
            Basal-ganglia action count.
        """
        super().__init__()
        self.vision = nn.Linear(n_input, spa_dim)
        self.wm_cell = nn.GRUCell(spa_dim, spa_dim)
        self.action = nn.Linear(spa_dim, n_actions)
        self.motor = nn.Linear(n_actions, 16)

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
        """Map visual input through simplified Spaun cognitive core.

        Parameters
        ----------
        img
            Flattened image tensor of shape ``(batch, 784)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Motor output and action probabilities.
        """
        vision = torch.tanh(self.vision(img))
        wm = self.wm_cell(vision, torch.zeros_like(vision))
        actions = torch.softmax(self.action(wm), dim=-1)
        motor = self.motor(actions)
        return motor, actions


def build() -> nn.Module:
    """Build a small simplified Spaun-core module.

    Returns
    -------
    nn.Module
        Configured ``SpaunCore`` instance.
    """
    return SpaunCore()


def example_input() -> Tensor:
    """Return a flattened image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 784)``.
    """
    return torch.randn(1, 784)
