"""Compact U-ViT diffusion backbone.

Bao et al., "All are Worth Words: A ViT Backbone for Diffusion Models", CVPR
2023.  U-ViT tokenizes all inputs including noisy image patches, timestep, and
condition tokens, processes them with a ViT, and uses long skip connections
between shallow and deep transformer layers instead of convolutional down/up
sampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t:
        Timestep tensor.
    dim:
        Embedding width.

    Returns
    -------
    torch.Tensor
        Timestep embeddings.
    """

    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
    emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class UViTCompact(nn.Module):
    """Small U-ViT with timestep token and long skip transformer connections."""

    def __init__(self, image_size: int = 32, patch: int = 8, dim: int = 64, depth: int = 4) -> None:
        """Initialize patch/token embeddings, transformer blocks, and unpatchify head.

        Parameters
        ----------
        image_size:
            Square image size.
        patch:
            Patch size.
        dim:
            Token width.
        depth:
            Number of transformer encoder layers.
        """

        super().__init__()
        self.patch = patch
        self.grid = image_size // patch
        self.to_patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.time = nn.Linear(dim, dim)
        self.pos = nn.Parameter(torch.randn(1, self.grid * self.grid + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation="gelu")
                for _ in range(depth)
            ]
        )
        self.skip_fuse = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(depth // 2)])
        self.to_pixels = nn.Linear(dim, 3 * patch * patch)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict diffusion noise using U-ViT tokens.

        Parameters
        ----------
        x:
            Noisy image.
        t:
            Diffusion timestep.

        Returns
        -------
        torch.Tensor
            Predicted image noise.
        """

        bsz = x.shape[0]
        patches = self.to_patch(x).flatten(2).transpose(1, 2)
        time = self.time(timestep_embedding(t, patches.shape[-1])).unsqueeze(1)
        tokens = torch.cat([time, patches], dim=1) + self.pos
        skips = []
        for index, block in enumerate(self.blocks):
            if index < len(self.blocks) // 2:
                tokens = block(tokens)
                skips.append(tokens)
            else:
                tokens = self.skip_fuse[index - len(self.blocks) // 2](
                    torch.cat([tokens, skips.pop()], dim=-1)
                )
                tokens = block(tokens)
        patches = self.to_pixels(tokens[:, 1:])
        patches = patches.view(bsz, self.grid, self.grid, 3, self.patch, self.patch)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            bsz, 3, self.grid * self.patch, self.grid * self.patch
        )


def build() -> nn.Module:
    """Build compact U-ViT diffusion backbone.

    Returns
    -------
    nn.Module
        U-ViT denoiser.
    """

    return UViTCompact()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy image and timestep inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Image and timestep.
    """

    return torch.randn(1, 3, 32, 32), torch.tensor([10])


MENAGERIE_ENTRIES = [
    (
        "simple_diffusion_uvit (all-inputs-as-tokens diffusion ViT with long skips)",
        "build",
        "example_input",
        "2023",
        "DC",
    ),
]
