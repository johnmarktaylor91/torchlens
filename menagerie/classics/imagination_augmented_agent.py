"""Imagination-Augmented Agent (I2A) for deep reinforcement learning.

Weber et al., 2017, "Imagination-Augmented Agents for Deep Reinforcement
Learning."  arXiv:1707.06203.

I2A combines a model-free policy pathway with model-based imagined rollouts.  A
learned environment model predicts future observations/rewards under candidate
rollout-policy actions; a rollout encoder summarizes each imagined trajectory;
the policy/value heads consume both the model-free observation embedding and the
imagination context.

This compact random-init version preserves the architecture-defining inference
graph while using small vector observations instead of Atari/Sokoban images:

  - observation encoder,
  - rollout policy producing branch action proposals,
  - environment model unrolled over several imagined branches and horizons,
  - GRU rollout encoder over predicted observation/reward features,
  - policy and value heads conditioned on model-free + imagined context.

Training objectives, sampled actions, and external environment calls are omitted;
candidate rollout branches are deterministic small action ids for traceability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentModel(nn.Module):
    """Compact learned one-step dynamics and reward model."""

    def __init__(self, latent_dim: int = 32, action_dim: int = 6, hidden: int = 48) -> None:
        """Initialize the transition and reward predictor.

        Parameters
        ----------
        latent_dim:
            Dimension of encoded observations.
        action_dim:
            Number of discrete actions.
        hidden:
            Hidden width for the environment model.
        """

        super().__init__()
        self.action_embed = nn.Embedding(action_dim, 8)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 8, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.next_latent = nn.Linear(hidden, latent_dim)
        self.reward = nn.Linear(hidden, 1)

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent observation and reward for one imagined step.

        Parameters
        ----------
        latent:
            Current latent observation with shape ``(batch, latent_dim)``.
        action:
            Discrete action ids with shape ``(batch,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Next latent observation and scalar reward prediction.
        """

        action_features = self.action_embed(action)
        hidden = self.net(torch.cat([latent, action_features], dim=-1))
        next_latent = latent + torch.tanh(self.next_latent(hidden))
        reward = self.reward(hidden)
        return next_latent, reward


class RolloutEncoder(nn.Module):
    """GRU encoder that summarizes imagined observation/reward rollouts."""

    def __init__(self, latent_dim: int = 32, hidden: int = 32) -> None:
        """Initialize the rollout recurrent encoder.

        Parameters
        ----------
        latent_dim:
            Dimension of imagined latent observations.
        hidden:
            Hidden width of the rollout summary GRU.
        """

        super().__init__()
        self.gru = nn.GRU(latent_dim + 1, hidden, batch_first=True)

    def forward(self, rollout: torch.Tensor) -> torch.Tensor:
        """Encode one imagined rollout sequence.

        Parameters
        ----------
        rollout:
            Sequence of concatenated next-latent and reward predictions with shape
            ``(batch, horizon, latent_dim + 1)``.

        Returns
        -------
        torch.Tensor
            Final rollout summary with shape ``(batch, hidden)``.
        """

        _, hidden = self.gru(rollout)
        return hidden[-1]


class ImaginationAugmentedAgent(nn.Module):
    """I2A policy/value network with learned imagined rollout context."""

    def __init__(
        self,
        obs_dim: int = 16,
        latent_dim: int = 32,
        action_dim: int = 6,
        branches: int = 3,
        horizon: int = 3,
    ) -> None:
        """Initialize compact I2A components.

        Parameters
        ----------
        obs_dim:
            Dimension of vector observations.
        latent_dim:
            Dimension of encoded observations.
        action_dim:
            Number of discrete actions.
        branches:
            Number of imagined rollout branches.
        horizon:
            Number of imagined steps per branch.
        """

        super().__init__()
        self.action_dim = action_dim
        self.branches = branches
        self.horizon = horizon
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
        )
        self.rollout_policy = nn.Linear(latent_dim, action_dim)
        self.env_model = EnvironmentModel(latent_dim, action_dim)
        self.rollout_encoder = RolloutEncoder(latent_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim + branches * 32, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim + branches * 32, 64), nn.ELU(), nn.Linear(64, 1)
        )

    def _branch_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Choose deterministic rollout branch actions from policy logits.

        Parameters
        ----------
        logits:
            Rollout-policy logits with shape ``(batch, action_dim)``.

        Returns
        -------
        torch.Tensor
            Branch action ids with shape ``(batch, branches)``.
        """

        probs = torch.softmax(logits, dim=-1)
        _, top_actions = torch.topk(probs, self.branches, dim=-1)
        return top_actions

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model-free and imagination-augmented policy/value inference.

        Parameters
        ----------
        obs:
            Vector observation with shape ``(batch, obs_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Policy logits and state-value estimate.
        """

        latent = self.obs_encoder(obs)
        branch_actions = self._branch_actions(self.rollout_policy(latent))
        summaries = []
        for branch in range(self.branches):
            imagined = latent
            action = branch_actions[:, branch]
            rollout_steps = []
            for _ in range(self.horizon):
                imagined, reward = self.env_model(imagined, action)
                rollout_steps.append(torch.cat([imagined, reward], dim=-1))
                action = self._branch_actions(self.rollout_policy(imagined))[:, 0]
            rollout = torch.stack(rollout_steps, dim=1)
            summaries.append(self.rollout_encoder(rollout))
        imagination = torch.cat(summaries, dim=-1)
        joint = torch.cat([latent, imagination], dim=-1)
        return self.policy_head(joint), self.value_head(joint)


def build() -> nn.Module:
    """Build the compact Imagination-Augmented Agent.

    Returns
    -------
    nn.Module
        Random-initialized compact I2A module.
    """

    return ImaginationAugmentedAgent()


def example_input() -> torch.Tensor:
    """Return a small vector observation batch.

    Returns
    -------
    torch.Tensor
        Example observation tensor with shape ``(1, 16)``.
    """

    return torch.randn(1, 16)


MENAGERIE_ENTRIES = [
    (
        "Imagination-Augmented Agent (learned rollout encoder policy)",
        "build",
        "example_input",
        "2017",
        "DC",
    ),
]
