"""QMIX monotonic value factorisation for multi-agent reinforcement learning.

Rashid et al., 2018, "QMIX: Monotonic Value Function Factorisation for Deep
Multi-Agent Reinforcement Learning."  arXiv:1803.11485.

QMIX trains decentralized per-agent action-value networks with a centralized
mixing network.  The distinctive architectural constraint is monotonicity:
``d Q_tot / d Q_a >= 0`` for every agent.  QMIX enforces this by generating
nonnegative mixing weights from the global state with hypernetworks, so argmax
over individual agent utilities remains consistent with the joint value.

This compact random-init module preserves the inference graph:

  - shared per-agent utility network from local observations to action values,
  - greedy per-agent utility selection,
  - state-conditioned hypernetworks that generate nonnegative mixer weights,
  - two-layer monotonic mixing network producing ``Q_tot``.

Replay buffers, target networks, recurrent hidden carry-over, and TD losses are
training machinery and are omitted from this forward-only classic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentUtilityNet(nn.Module):
    """Shared decentralized per-agent utility network."""

    def __init__(self, obs_dim: int = 8, hidden: int = 32, n_actions: int = 5) -> None:
        """Initialize the shared agent Q-network.

        Parameters
        ----------
        obs_dim:
            Local observation dimension for each agent.
        hidden:
            Hidden width for the utility network.
        n_actions:
            Number of discrete actions per agent.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute per-agent action values.

        Parameters
        ----------
        obs:
            Local observations with shape ``(batch, n_agents, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Action values with shape ``(batch, n_agents, n_actions)``.
        """

        batch, n_agents, obs_dim = obs.shape
        q_values = self.net(obs.reshape(batch * n_agents, obs_dim))
        return q_values.reshape(batch, n_agents, -1)


class QMixer(nn.Module):
    """State-conditioned monotonic QMIX mixing network."""

    def __init__(self, n_agents: int = 3, state_dim: int = 12, embed_dim: int = 16) -> None:
        """Initialize hypernetworks for nonnegative mixer weights.

        Parameters
        ----------
        n_agents:
            Number of decentralized agents.
        state_dim:
            Global state dimension available during centralized training/inference.
        embed_dim:
            Hidden width of the mixing network.
        """

        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix individual utilities into a monotonic joint value.

        Parameters
        ----------
        agent_qs:
            Selected per-agent utilities with shape ``(batch, n_agents)``.
        state:
            Global state tensor with shape ``(batch, state_dim)``.

        Returns
        -------
        torch.Tensor
            Joint action value ``Q_tot`` with shape ``(batch, 1)``.
        """

        batch = agent_qs.shape[0]
        w1 = self.hyper_w1(state).abs().reshape(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).reshape(batch, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = self.hyper_w2(state).abs().reshape(batch, self.embed_dim, 1)
        v = self.value(state).reshape(batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + v
        return q_tot.squeeze(1)


class QMIX(nn.Module):
    """Compact QMIX agent utilities plus monotonic mixer."""

    def __init__(self, n_agents: int = 3, obs_dim: int = 8, state_dim: int = 12) -> None:
        """Initialize compact QMIX.

        Parameters
        ----------
        n_agents:
            Number of decentralized agents.
        obs_dim:
            Per-agent local observation dimension.
        state_dim:
            Global state dimension for the mixer hypernetworks.
        """

        super().__init__()
        self.agent = AgentUtilityNet(obs_dim=obs_dim)
        self.mixer = QMixer(n_agents=n_agents, state_dim=state_dim)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-agent utilities and monotonic joint value.

        Parameters
        ----------
        inputs:
            Tuple ``(obs, state)`` where ``obs`` is ``(batch, n_agents, obs_dim)``
            and ``state`` is ``(batch, state_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Joint value ``Q_tot`` and all per-agent action values.
        """

        obs, state = inputs
        q_values = self.agent(obs)
        selected_qs = q_values.max(dim=-1).values
        q_tot = self.mixer(selected_qs, state)
        return q_tot, q_values


def build() -> nn.Module:
    """Build the compact QMIX model.

    Returns
    -------
    nn.Module
        Random-initialized compact QMIX module.
    """

    return QMIX()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return synthetic local observations and centralized state.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(obs, state)`` tensors for three agents.
    """

    return torch.randn(1, 3, 8), torch.randn(1, 12)


MENAGERIE_ENTRIES = [
    (
        "QMIX (monotonic value factorisation mixer)",
        "build",
        "example_input",
        "2018",
        "DC",
    ),
]
