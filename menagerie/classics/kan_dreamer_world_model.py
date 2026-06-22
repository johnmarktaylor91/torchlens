"""KAN-Dreamer compact world model.

Paper: Shi and Luan, 2025, "KAN-Dreamer: Benchmarking Kolmogorov-Arnold
Networks as Function Approximators in World Models".

KAN-Dreamer replaces selected DreamerV3 MLP/CNN components with KAN/FastKAN
layers.  This compact world model keeps recurrent latent prediction plus
FastKAN reward/continue/value heads with learnable radial-basis edge functions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FastKANLayer(nn.Module):
    """FastKAN layer using RBF basis functions on each input dimension."""

    def __init__(self, in_features: int, out_features: int, grid: int = 5) -> None:
        """Initialize RBF centers and edge weights."""

        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-1, 1, grid), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(in_features, grid))
        self.weight = nn.Parameter(torch.randn(in_features, grid, out_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate learnable edge functions and sum them into outputs."""

        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) * F.softplus(self.scale))
        return torch.einsum("...ig,igo->...o", basis, self.weight) + self.bias


class KANDreamerWorldModel(nn.Module):
    """Compact Dreamer-style latent dynamics with FastKAN predictors."""

    def __init__(self, obs_dim: int = 12, action_dim: int = 4, latent_dim: int = 24) -> None:
        """Initialize encoder, recurrent dynamics, and KAN heads."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim), nn.ELU(), FastKANLayer(latent_dim, latent_dim)
        )
        self.gru = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.prior = FastKANLayer(latent_dim, latent_dim)
        self.reward = nn.Sequential(FastKANLayer(latent_dim, 32), nn.SiLU(), FastKANLayer(32, 1))
        self.continue_head = nn.Sequential(
            FastKANLayer(latent_dim, 32), nn.SiLU(), FastKANLayer(32, 1)
        )
        self.value = nn.Sequential(FastKANLayer(latent_dim, 32), nn.SiLU(), FastKANLayer(32, 1))

    def forward(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Roll a compact recurrent state-space world model."""

        hidden = torch.zeros(obs.shape[0], 24, device=obs.device)
        rewards = []
        continues = []
        values = []
        for step in range(obs.shape[1]):
            embed = self.encoder(obs[:, step])
            hidden = self.gru(torch.cat([embed, actions[:, step]], dim=-1), hidden)
            latent = torch.tanh(self.prior(hidden))
            rewards.append(self.reward(latent))
            continues.append(torch.sigmoid(self.continue_head(latent)))
            values.append(self.value(latent))
        return torch.stack(rewards, 1), torch.stack(continues, 1), torch.stack(values, 1)


def build() -> nn.Module:
    """Build the compact KAN-Dreamer world model."""

    return KANDreamerWorldModel().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return observation and action sequences."""

    return torch.randn(1, 5, 12), torch.randn(1, 5, 4)


MENAGERIE_ENTRIES = [
    ("kan_dreamer_world_model", "build", "example_input", "2025", "RL"),
]
