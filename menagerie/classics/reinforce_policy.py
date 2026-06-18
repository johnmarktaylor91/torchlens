"""REINFORCE policy network, 1992, Williams.

Paper: "Simple statistical gradient-following algorithms for connectionist reinforcement learning."
The module returns policy logits, probabilities, and log-probabilities; reward-weighted
gradient estimation and stochastic sampling are intentionally outside the traceable forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class REINFORCEPolicyNetwork(nn.Module):
    """Small stochastic-policy substrate for categorical actions."""

    def __init__(self, n_state: int = 16, hidden_size: int = 24, n_actions: int = 4) -> None:
        """Initialize the policy MLP.

        Parameters
        ----------
        n_state
            Number of state features.
        hidden_size
            Number of hidden units.
        n_actions
            Number of action logits.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute policy logits, probabilities, and log-probabilities.

        Parameters
        ----------
        state
            State tensor of shape ``(batch, n_state)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Logits, categorical probabilities, and log-probabilities.
        """
        logits = self.net(state)
        return logits, torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)


def build() -> nn.Module:
    """Build a small REINFORCE policy network.

    Returns
    -------
    nn.Module
        Configured ``REINFORCEPolicyNetwork`` instance.
    """
    return REINFORCEPolicyNetwork()


def example_input() -> Tensor:
    """Create an example state vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 16)``.
    """
    return torch.randn(1, 16)


MENAGERIE_ENTRIES = [
    ("REINFORCE Policy Network (Williams)", "build", "example_input", "1992", "DD")
]
