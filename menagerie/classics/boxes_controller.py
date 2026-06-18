"""BOXES adaptive controller, 1968, Michie and Chambers.

Paper: "BOXES: An experiment in adaptive control."
Continuous state is discretized into boxes whose action propensities are read as
controller logits; delayed reinforcement updates to those propensities are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BOXESAdaptiveController(nn.Module):
    """Discretized-state lookup controller with learned action logits."""

    def __init__(self, n_dims: int = 4, n_bins: int = 4, n_actions: int = 3) -> None:
        """Initialize box boundaries and action propensities.

        Parameters
        ----------
        n_dims
            Number of continuous state dimensions.
        n_bins
            Number of bins per dimension.
        n_actions
            Number of controller actions.
        """
        super().__init__()
        self.n_bins = n_bins
        num_boxes = n_bins**n_dims
        self.logits = nn.Parameter(torch.randn(num_boxes, n_actions) * 0.05)
        edges = torch.linspace(-1.0, 1.0, n_bins + 1)[1:-1]
        multipliers = torch.tensor([n_bins**i for i in range(n_dims)], dtype=torch.long)
        self.register_buffer("edges", edges)
        self.register_buffer("multipliers", multipliers)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Look up action propensities for discretized state boxes.

        Parameters
        ----------
        state
            Continuous state tensor of shape ``(batch, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Box-specific action logits and probabilities.
        """
        digits = (state.unsqueeze(-1) > self.edges).to(torch.long).sum(dim=-1)
        box_id = (digits * self.multipliers).sum(dim=-1)
        logits = self.logits.index_select(0, box_id)
        return logits, torch.softmax(logits, dim=-1)


def build() -> nn.Module:
    """Build a compact BOXES controller.

    Returns
    -------
    nn.Module
        Configured ``BOXESAdaptiveController`` instance.
    """
    return BOXESAdaptiveController()


def example_input() -> Tensor:
    """Create an example continuous controller state.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.tensor([[0.2, -0.7, 0.4, 0.9]])


MENAGERIE_ENTRIES = [("BOXES Adaptive Controller", "build", "example_input", "1968", "DD")]
