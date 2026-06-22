"""Plan2Explore self-supervised world-model agent.

Sekar et al. (ICML 2020), "Planning to Explore via Self-Supervised World
Models."  Plan2Explore learns a Dreamer/PlaNet-style latent world model, uses an
ensemble of latent dynamics heads to estimate disagreement/information gain, and
trains an exploration actor to maximize that intrinsic reward before task
adaptation.  This compact reconstruction keeps the visual encoder, recurrent
latent dynamics, ensemble disagreement, actor, and value head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactPlan2Explore(nn.Module):
    """Compact Plan2Explore inference graph."""

    def __init__(self, actions: int = 5, latent: int = 32, ensemble: int = 4) -> None:
        """Initialize the compact agent.

        Parameters
        ----------
        actions:
            Number of discrete actions.
        latent:
            Latent state size.
        ensemble:
            Number of disagreement dynamics heads.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.ELU(inplace=False),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ELU(inplace=False),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, latent),
            nn.LayerNorm(latent),
        )
        self.action_embed = nn.Embedding(actions, latent)
        self.rnn = nn.GRUCell(latent * 2, latent)
        self.ensemble = nn.ModuleList([nn.Linear(latent, latent) for _ in range(ensemble)])
        self.actor = nn.Linear(latent, actions)
        self.value = nn.Linear(latent, 1)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Roll a latent world model and compute intrinsic disagreement.

        Parameters
        ----------
        frames:
            Observation tensor of shape ``(B, T, 3, 32, 32)``.
        actions:
            Discrete action tensor of shape ``(B, T)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Actor logits, value predictions, and disagreement reward.
        """

        batch, steps = actions.shape
        state = frames.new_zeros(batch, self.rnn.hidden_size)
        states = []
        disagreements = []
        for t in range(steps):
            obs = self.encoder(frames[:, t])
            act = self.action_embed(actions[:, t])
            state = self.rnn(torch.cat([obs, act], dim=-1), state)
            preds = torch.stack([head(state) for head in self.ensemble], dim=0)
            disagreements.append(preds.var(dim=0).mean(dim=-1))
            states.append(state)
        latent_states = torch.stack(states, dim=1)
        return {
            "policy": self.actor(latent_states),
            "value": self.value(latent_states).squeeze(-1),
            "intrinsic_reward": torch.stack(disagreements, dim=1),
        }


def build() -> nn.Module:
    """Build the compact Plan2Explore agent.

    Returns
    -------
    nn.Module
        Random-init Plan2Explore model in evaluation mode.
    """

    return CompactPlan2Explore().eval()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact image observations and actions.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Frames ``(1, 4, 3, 32, 32)`` and actions ``(1, 4)``.
    """

    return torch.randn(1, 4, 3, 32, 32), torch.randint(0, 5, (1, 4))


MENAGERIE_ENTRIES = [
    ("Plan2Explore", "build", "example_input", "2020", "E5"),
]
