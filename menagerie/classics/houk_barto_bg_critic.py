"""Houk-Adams-Barto basal-ganglia critic, 1995, Houk, Adams, and Barto.

Paper: "A model of how the basal ganglia generate and use neural signals that predict
reinforcement." Corticostriatal actor logits and a critic value head produce a
dopamine-like TD-error proxy from the current state.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Houk-Adams-Barto Basal Ganglia Critic", "build", "example_input", "1995", "DB")
]


class HoukBartoBGCritic(nn.Module):
    """Actor-critic basal-ganglia loop with pathway masks."""

    def __init__(self, n_state: int = 32, n_hidden: int = 32, n_actions: int = 6) -> None:
        """Initialize cortical encoder, actor, and critic heads.

        Parameters
        ----------
        n_state
            Cortical state dimensionality.
        n_hidden
            Striatal feature dimensionality.
        n_actions
            Number of actions.
        """
        super().__init__()
        self.encoder = nn.Linear(n_state, n_hidden)
        self.actor = nn.Linear(n_hidden, n_actions)
        self.critic = nn.Linear(n_hidden, 1)
        mask = torch.ones(n_actions)
        mask[1::2] = -1.0
        self.register_buffer("pathway_mask", mask)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute actor logits, value, and dopamine proxy.

        Parameters
        ----------
        state
            State tensor of shape ``(batch, n_state)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Masked action logits, critic value, and bounded TD-error proxy.
        """
        hidden = torch.relu(self.encoder(state))
        logits = self.actor(hidden) * self.pathway_mask
        value = self.critic(hidden)
        td_error = torch.tanh(value)
        return logits, value, td_error


def build() -> nn.Module:
    """Build a small Houk-Adams-Barto BG critic module.

    Returns
    -------
    nn.Module
        Configured ``HoukBartoBGCritic`` instance.
    """
    return HoukBartoBGCritic()


def example_input() -> Tensor:
    """Return a cortical state example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 32)``.
    """
    return torch.randn(1, 32)
