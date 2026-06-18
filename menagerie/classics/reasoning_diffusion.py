"""Diffusion-of-Thought, 2024, latent reasoning diffusion.

Paper: 2024, "Diffusion of Thought."
A transformer denoiser predicts cleaner reasoning latents from noisy latent chains; this
minimal module omits the text decoder, sampler, and training loss schedule.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiffusionOfThought(nn.Module):
    """Transformer denoiser over reasoning-latent sequences."""

    def __init__(self, dim: int = 256, n_heads: int = 4) -> None:
        """Initialize time embedding, transformer block, and noise head.

        Parameters
        ----------
        dim
            Reasoning latent width.
        n_heads
            Number of attention heads.
        """
        super().__init__()
        self.time_scale = nn.Parameter(torch.randn(dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(dim, dim)

    def forward(self, latents: Tensor) -> Tensor:
        """Predict denoised reasoning latents.

        Parameters
        ----------
        latents
            Reasoning latents with shape ``(batch, 32, 256)``.

        Returns
        -------
        Tensor
            Predicted clean latents.
        """
        steps = torch.linspace(
            0.0, 1.0, latents.shape[1], device=latents.device, dtype=latents.dtype
        )
        x = latents + steps.reshape(1, -1, 1) * self.time_scale
        return self.head(self.encoder(x))


MENAGERIE_ENTRIES = [("Diffusion-of-Thought", "build", "example_input", "2024", "DA")]


def build() -> nn.Module:
    """Build a Diffusion-of-Thought denoiser.

    Returns
    -------
    nn.Module
        Configured denoiser module.
    """
    return DiffusionOfThought()


def example_input() -> Tensor:
    """Create reasoning latents.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32, 256)``.
    """
    return torch.randn(1, 32, 256)
