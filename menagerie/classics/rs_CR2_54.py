# FAITHFUL REIMPLEMENTATION from Science Robotics DOI:10.1126/scirobotics.adi7566 (no public code)
from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "reimpl-pytorch"


class SkillPolicy(nn.Module):
    """Low-level locomotion skill policy."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """Initialize a compact skill controller."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 24), nn.ELU(), nn.Linear(24, action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict joint commands for one terrain skill."""
        return self.net(obs)


class ANYmalParkour(nn.Module):
    """Hierarchical parkour policy selecting and controlling locomotion skills."""

    def __init__(
        self, obs_dim: int = 10, terrain_dim: int = 12, action_dim: int = 8, skills: int = 5
    ) -> None:
        """Initialize terrain encoder, high-level selector, and skill catalog."""
        super().__init__()
        self.terrain_encoder = nn.Sequential(
            nn.Linear(terrain_dim, 24), nn.ELU(), nn.Linear(24, 16), nn.ELU()
        )
        self.selector = nn.Linear(obs_dim + 16, skills)
        self.skill_condition = nn.Linear(skills, obs_dim)
        self.skills = nn.ModuleList([SkillPolicy(obs_dim, action_dim) for _ in range(skills)])

    def forward(
        self, proprioception: torch.Tensor, terrain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select among pretrained skills and blend their low-level actions."""
        terrain_code = self.terrain_encoder(terrain)
        logits = self.selector(torch.cat((proprioception, terrain_code), dim=-1))
        weights = torch.softmax(logits, dim=-1)
        conditioned = proprioception + self.skill_condition(weights)
        actions = torch.stack([skill(conditioned) for skill in self.skills], dim=1)
        return (actions * weights.unsqueeze(-1)).sum(dim=1), logits


def build_anymal_parkour() -> ANYmalParkour:
    """Build the toy ANYmal Parkour hierarchy."""
    return ANYmalParkour()


def example_input_anymal_parkour() -> tuple[torch.Tensor, torch.Tensor]:
    """Return toy proprioception and reconstructed terrain features."""
    return torch.randn(1, 10), torch.randn(1, 12)


MENAGERIE_ENTRIES = [
    ("ANYmal Parkour", build_anymal_parkour, example_input_anymal_parkour, 2024, "REIMPL")
]
