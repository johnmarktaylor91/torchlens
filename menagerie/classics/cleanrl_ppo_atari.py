"""CleanRL PPO Atari convolutional actor-critic.

CleanRL's Atari PPO script follows the canonical Nature-DQN convolutional torso
for frame stacks, orthogonal initialization, separate actor and critic heads, and
PPO's categorical policy/value forward path.  This module captures the model
architecture without depending on Gym/ALE or the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _orthogonal(layer: nn.Module, gain: float) -> nn.Module:
    """Apply CleanRL-style orthogonal initialization to a layer.

    Parameters
    ----------
    layer:
        Linear or convolutional layer.
    gain:
        Orthogonal initialization gain.

    Returns
    -------
    nn.Module
        The initialized layer.
    """

    if isinstance(layer, nn.Conv2d | nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class CleanRLPPOAtari(nn.Module):
    """Atari PPO policy/value network from CleanRL."""

    def __init__(self, actions: int = 6) -> None:
        """Initialize the shared CNN torso and actor/critic heads.

        Parameters
        ----------
        actions:
            Number of discrete Atari actions.
        """

        super().__init__()
        self.network = nn.Sequential(
            _orthogonal(nn.Conv2d(4, 32, 8, stride=4), 2**0.5),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(32, 64, 4, stride=2), 2**0.5),
            nn.ReLU(),
            _orthogonal(nn.Conv2d(64, 64, 3, stride=1), 2**0.5),
            nn.ReLU(),
            nn.Flatten(),
            _orthogonal(nn.Linear(64 * 7 * 7, 512), 2**0.5),
            nn.ReLU(),
        )
        self.actor = _orthogonal(nn.Linear(512, actions), 0.01)
        self.critic = _orthogonal(nn.Linear(512, 1), 1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and value for frame stacks.

        Parameters
        ----------
        x:
            Atari frame stack ``(B, 4, 84, 84)`` with byte-like scale.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Categorical policy logits and scalar value.
        """

        hidden = self.network(x / 255.0)
        return self.actor(hidden), self.critic(hidden)


def build() -> nn.Module:
    """Build the CleanRL PPO Atari network.

    Returns
    -------
    nn.Module
        Actor-critic model.
    """

    return CleanRLPPOAtari()


def example_input() -> torch.Tensor:
    """Create a small Atari frame-stack input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 4, 84, 84)``.
    """

    return torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.float32)


MENAGERIE_ENTRIES = [
    (
        "CleanRL PPO Atari (Nature-CNN categorical actor-critic)",
        "build",
        "example_input",
        "2022",
        "DC",
    ),
]
