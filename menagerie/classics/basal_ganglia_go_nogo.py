"""Basal ganglia Go/NoGo actor-critic, 1996, Cohen and Frank lineage.

Paper: "A model of dopamine and basal ganglia interactions in sequence learning."
Separate Go and NoGo action channels compete through their difference while a critic
estimates value; dopamine update rules are outside the forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Basal Ganglia Go/NoGo Actor-Critic (OpAL)", "build", "example_input", "1996", "DB")
]


class BasalGangliaGoNoGo(nn.Module):
    """Go/NoGo actor with critic value head."""

    def __init__(self, n_state: int = 32, n_hidden: int = 32, n_actions: int = 6) -> None:
        """Initialize state encoder, Go/NoGo heads, and critic.

        Parameters
        ----------
        n_state
            State-vector dimensionality.
        n_hidden
            Encoded corticostriatal feature size.
        n_actions
            Number of action channels.
        """
        super().__init__()
        self.encoder = nn.Linear(n_state, n_hidden)
        self.go = nn.Linear(n_hidden, n_actions)
        self.nogo = nn.Linear(n_hidden, n_actions)
        self.critic = nn.Linear(n_hidden, 1)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Compute action probabilities and state value.

        Parameters
        ----------
        state
            State tensor of shape ``(batch, n_state)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Softmax action probabilities and critic value.
        """
        hidden = torch.relu(self.encoder(state))
        logits = self.go(hidden) - self.nogo(hidden)
        return torch.softmax(logits, dim=-1), self.critic(hidden)


def build() -> nn.Module:
    """Build a small basal-ganglia Go/NoGo module.

    Returns
    -------
    nn.Module
        Configured ``BasalGangliaGoNoGo`` instance.
    """
    return BasalGangliaGoNoGo()


def example_input() -> Tensor:
    """Return a state-vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 32)``.
    """
    return torch.randn(1, 32)
