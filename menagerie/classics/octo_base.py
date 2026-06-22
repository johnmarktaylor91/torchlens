"""Octo transformer-based diffusion robot policy.

Paper: Octo Model Team 2024, "Octo: An Open-Source Generalist Robot Policy."
Octo tokenizes image observations and language/task inputs, processes flexible
tokens with a Transformer, and decodes action chunks with a diffusion head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OctoPolicy(nn.Module):
    """Compact Octo policy with ViT tokens and diffusion action head."""

    def __init__(self, width: int = 48, layers: int = 2, action_dim: int = 7) -> None:
        """Initialize Octo components.

        Parameters
        ----------
        width:
            Transformer width.
        layers:
            Number of Transformer encoder layers.
        action_dim:
            Action dimension per chunk step.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, width, 8, stride=8)
        self.lang = nn.Embedding(128, width)
        self.proprio = nn.Linear(action_dim, width)
        self.noisy_action = nn.Linear(action_dim + 1, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 3, batch_first=True)
        self.trunk = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, action_dim)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Predict denoising vectors for an action chunk.

        Parameters
        ----------
        image:
            RGB observation.
        text:
            Language token ids.
        proprio:
            Robot state vector.
        noisy_action:
            Noisy action chunk.
        sigma:
            Diffusion noise level.

        Returns
        -------
        torch.Tensor
            Action denoising vectors.
        """

        img = self.patch(image).flatten(2).transpose(1, 2)
        lang = self.lang(text)
        prop = self.proprio(proprio).unsqueeze(1)
        sig = sigma.view(sigma.shape[0], 1, 1).expand(-1, noisy_action.shape[1], 1)
        act = self.noisy_action(torch.cat([noisy_action, sig], dim=-1))
        tokens = self.trunk(torch.cat([img, lang, prop, act], dim=1))
        return self.head(tokens[:, -noisy_action.shape[1] :])


def build() -> nn.Module:
    """Build Octo base.

    Returns
    -------
    nn.Module
        Octo policy.
    """

    return OctoPolicy(width=48, layers=2)


def build_small() -> nn.Module:
    """Build Octo small.

    Returns
    -------
    nn.Module
        Smaller Octo policy.
    """

    return OctoPolicy(width=32, layers=1)


def build_tiny() -> nn.Module:
    """Build Octo tiny.

    Returns
    -------
    nn.Module
        Tiny Octo policy.
    """

    return OctoPolicy(width=24, layers=1)


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Octo policy inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Image, text, proprioception, noisy actions, and sigma.
    """

    return (
        torch.randn(1, 3, 32, 32),
        torch.randint(0, 128, (1, 6)),
        torch.randn(1, 7),
        torch.randn(1, 4, 7),
        torch.ones(1),
    )


MENAGERIE_ENTRIES = [
    ("octo_base", "build", "example_input", "2024", "E7"),
    ("Octo_base", "build", "example_input", "2024", "E7"),
    ("octo_small", "build_small", "example_input", "2024", "E7"),
    ("Octo_small", "build_small", "example_input", "2024", "E7"),
    ("Octo_tiny", "build_tiny", "example_input", "2024", "E7"),
]
