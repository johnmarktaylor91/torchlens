"""OpenAI Point-E upsampler: coarse-to-fine point-cloud diffusion.

Paper: Nichol et al. 2022, "Point-E: A System for Generating 3D Point Clouds
from Complex Prompts".

The compact module keeps the Point-E upsampler's conditioning pattern: a noisy
fine point set attends to an already sampled coarse point cloud before predicting
fine coordinate residuals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointEUpsampler(nn.Module):
    """Compact cross-attention point-cloud upsampler."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize point encoders and cross-attention upsampling blocks.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.fine_in = nn.Linear(3, dim)
        self.coarse_in = nn.Linear(3, dim)
        self.cross = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.refine = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, dim)
        )
        self.out = nn.Linear(dim, 3)

    def forward(self, noisy_fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """Predict fine point residuals conditioned on coarse points.

        Parameters
        ----------
        noisy_fine:
            Noisy high-resolution point set.
        coarse:
            Coarse conditioning point set.

        Returns
        -------
        torch.Tensor
            Fine coordinate residuals.
        """

        fine = self.fine_in(noisy_fine)
        coarse_tokens = self.coarse_in(coarse)
        attn, _ = self.cross(fine, coarse_tokens, coarse_tokens)
        hidden = fine + attn
        hidden = hidden + self.refine(hidden)
        return self.out(hidden)


def build() -> nn.Module:
    """Build the compact Point-E upsampler.

    Returns
    -------
    nn.Module
        Random-initialized upsampler.
    """

    return PointEUpsampler()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create fine and coarse point clouds.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Noisy fine points and coarse conditioning points.
    """

    return torch.randn(1, 24, 3), torch.randn(1, 8, 3)


MENAGERIE_ENTRIES = [
    ("openai_point_e_upsampler", "build", "example_input", "2022", "E6"),
]
