"""OpenAI Point-E prior: text/image conditioned point-cloud diffusion.

Paper: Nichol et al. 2022, "Point-E: A System for Generating 3D Point Clouds
from Complex Prompts", arXiv:2212.08751.

This compact prior mirrors the first point-cloud diffusion stage: noisy point
tokens are processed by a transformer conditioned on prompt/image embeddings,
with a coordinate denoising head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointEPrior(nn.Module):
    """Compact Point-E point-cloud diffusion prior."""

    def __init__(self, points: int = 16, dim: int = 32) -> None:
        """Initialize point, timestep, and prompt-conditioned transformer layers.

        Parameters
        ----------
        points:
            Number of point tokens.
        dim:
            Transformer width.
        """

        super().__init__()
        self.point_in = nn.Linear(3, dim)
        self.prompt = nn.Embedding(64, dim)
        self.time = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.point_out = nn.Linear(dim, 3)
        self.points = points

    def forward(
        self, noisy_points: torch.Tensor, tokens: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict denoised point-cloud coordinates.

        Parameters
        ----------
        noisy_points:
            Noisy point coordinates.
        tokens:
            Prompt token ids.
        t:
            Diffusion timestep/noise level.

        Returns
        -------
        torch.Tensor
            Coordinate residuals for each point.
        """

        point_tokens = self.point_in(noisy_points)
        cond = self.prompt(tokens).mean(dim=1, keepdim=True) + self.time(t).unsqueeze(1)
        hidden = self.transformer(point_tokens + cond)
        return self.point_out(hidden)


def build() -> nn.Module:
    """Build the compact Point-E prior.

    Returns
    -------
    nn.Module
        Random-initialized Point-E prior.
    """

    return PointEPrior()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create noisy point tokens, text ids, and a diffusion time.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Example prior inputs.
    """

    return torch.randn(1, 16, 3), torch.randint(0, 64, (1, 5)), torch.ones(1, 1) * 0.5


MENAGERIE_ENTRIES = [
    ("openai_point_e_prior", "build", "example_input", "2022", "E6"),
]
