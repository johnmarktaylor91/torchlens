"""ASE-ACE adaptive critic, 1983, Barto, Sutton, and Anderson.

Paper: "Neuronlike adaptive elements that can solve difficult learning control problems."
The forward pass exposes the trace-clean actor logit/probability and critic value
substrate; stochastic action sampling and eligibility-trace TD updates are training-loop
mechanisms omitted here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ASEACEAdaptiveCritic(nn.Module):
    """Small actor-critic substrate with separate ASE actor and ACE critic."""

    def __init__(self, n_feat: int = 32) -> None:
        """Initialize actor and critic linear elements.

        Parameters
        ----------
        n_feat
            Number of state features.
        """
        super().__init__()
        self.actor = nn.Linear(n_feat, 1)
        self.critic = nn.Linear(n_feat, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute actor probability, critic value, and raw actor logit.

        Parameters
        ----------
        x
            State feature tensor of shape ``(batch, n_feat)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Actor Bernoulli probability, ACE value estimate, and ASE logit.
        """
        logit = self.actor(x)
        value = self.critic(x)
        action_prob = torch.sigmoid(logit)
        return action_prob, value, logit


def build() -> nn.Module:
    """Build a small random-init ASE-ACE adaptive critic.

    Returns
    -------
    nn.Module
        Configured ``ASEACEAdaptiveCritic`` instance.
    """
    return ASEACEAdaptiveCritic()


def example_input() -> Tensor:
    """Create an example state vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32)``.
    """
    return torch.randn(1, 32)


MENAGERIE_ENTRIES = [
    ("ASE-ACE Adaptive Critic (Barto-Sutton-Anderson)", "build", "example_input", "1983", "DD")
]
