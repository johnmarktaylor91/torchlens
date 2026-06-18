"""GARIC fuzzy actor-critic controller, 1992, Berenji and Khedkar.

Paper: "Learning and tuning fuzzy logic controllers through reinforcements."
Gaussian rule firing strengths blend consequent action preferences while a separate
critic estimates value; reinforcement-based rule tuning is outside this forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GARICFuzzyActorCriticController(nn.Module):
    """Gaussian fuzzy actor with an MLP critic."""

    def __init__(self, n_state: int = 4, n_rules: int = 6, n_actions: int = 2) -> None:
        """Initialize fuzzy rule parameters and critic.

        Parameters
        ----------
        n_state
            Number of input state variables.
        n_rules
            Number of fuzzy rules.
        n_actions
            Number of action channels.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, n_rules)[:, None].repeat(1, n_state))
        self.log_scales = nn.Parameter(torch.zeros(n_rules, n_state))
        self.consequents = nn.Parameter(torch.randn(n_rules, n_actions) * 0.2)
        self.critic = nn.Sequential(nn.Linear(n_state, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute fuzzy actor output and critic value.

        Parameters
        ----------
        state
            Input state tensor of shape ``(batch, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Blended action, critic value, and normalized rule firing strengths.
        """
        scales = torch.exp(self.log_scales) + 0.05
        normalized = (state[:, None, :] - self.centers[None, :, :]) / scales[None, :, :]
        firing = torch.exp(-0.5 * (normalized * normalized).sum(dim=-1))
        rule_weights = firing / firing.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        action = rule_weights @ self.consequents
        value = self.critic(state)
        return action, value, rule_weights


def build() -> nn.Module:
    """Build a GARIC fuzzy actor-critic controller.

    Returns
    -------
    nn.Module
        Configured ``GARICFuzzyActorCriticController`` instance.
    """
    return GARICFuzzyActorCriticController()


def example_input() -> Tensor:
    """Create an example fuzzy-controller state.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [
    ("GARIC Fuzzy Actor-Critic Controller", "build", "example_input", "1992", "DD")
]
