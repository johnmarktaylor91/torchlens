# FAITHFUL REIMPLEMENTATION from https://arxiv.org/abs/2412.17394 (no public code)
"""Toy AeroDiT latent diffusion transformer for airfoil flow fields."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class DiTBlock(nn.Module):
    """Diffusion Transformer block with timestep conditioning."""

    def __init__(self, dim: int, heads: int) -> None:
        """Initialize a DiT block.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention heads.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, tokens: Tensor, cond: Tensor) -> Tensor:
        """Apply conditioned self-attention and feed-forward updates.

        Parameters
        ----------
        tokens:
            Patch tokens.
        cond:
            Conditioning embedding.

        Returns
        -------
        Tensor
            Updated tokens.
        """
        shift, scale = self.condition(cond).chunk(2, dim=-1)
        normalized = self.norm1(tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class AeroDiT(nn.Module):
    """Diffusion Transformer denoiser for RANS airfoil flow fields."""

    def __init__(self, channels: int = 3, dim: int = 32, patch: int = 4) -> None:
        """Initialize AeroDiT.

        Parameters
        ----------
        channels:
            Flow-field channels.
        dim:
            Transformer width.
        patch:
            Patch size.
        """
        super().__init__()
        self.channels = channels
        self.patch = patch
        self.patch_embed = nn.Linear(channels * patch * patch, dim)
        self.condition = nn.Linear(4, dim)
        self.time = nn.Linear(2, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, 4), DiTBlock(dim, 4)])
        self.output = nn.Linear(dim, channels * patch * patch)

    def forward(self, sample: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Predict denoised flow-field residuals.

        Parameters
        ----------
        sample:
            Noisy flow field, airfoil/flow condition vector, and diffusion timestep.

        Returns
        -------
        Tensor
            Denoised flow field prediction.
        """
        flow, condition, timestep = sample
        batch, channels, height, width = flow.shape
        patches = flow.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch, -1, channels * self.patch * self.patch
        )
        tokens = self.patch_embed(patches)
        time_features = torch.cat(
            [torch.sin(timestep[:, None] * math.pi), torch.cos(timestep[:, None] * math.pi)], dim=-1
        )
        cond = self.condition(condition) + self.time(time_features)
        for block in self.blocks:
            tokens = block(tokens, cond)
        out = self.output(tokens).view(
            batch, height // self.patch, width // self.patch, channels, self.patch, self.patch
        )
        return out.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


def build_aerodit() -> AeroDiT:
    """Build a tiny AeroDiT model.

    Returns
    -------
    AeroDiT
        Model instance.
    """
    return AeroDiT()


def example_input_aerodit() -> tuple[Tensor, Tensor, Tensor]:
    """Create example flow field, conditioning features, and timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Noisy flow, condition vector, and timestep.
    """
    return torch.randn(2, 3, 16, 16), torch.randn(2, 4), torch.tensor([0.2, 0.7])


MENAGERIE_ENTRIES = [("AeroDiT", build_aerodit, example_input_aerodit, 2024, "REIMPL")]
