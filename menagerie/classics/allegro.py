"""Allegro text-to-video model: compact 3D Diffusion Transformer core.

Rhymes AI Allegro (2024) is described as a commercial-level open text-to-video
model using a VideoVAE latent space and a VideoDiT backbone with 3D positional
encoding/full attention conditioned on text. This module reconstructs the
dependency-gated core as a small random-init latent video diffusion transformer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm modulation from timestep/text conditioning."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        """Initialize normalization and modulation projection.

        Parameters
        ----------
        dim:
            Token feature dimension.
        cond_dim:
            Conditioning vector dimension.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mod = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply condition-dependent shift and scale.

        Parameters
        ----------
        x:
            Token tensor.
        cond:
            Conditioning tensor.

        Returns
        -------
        Tensor
            Modulated normalized tokens.
        """
        shift, scale = self.mod(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class AllegroBlock(nn.Module):
    """VideoDiT transformer block with adaptive normalization."""

    def __init__(self, dim: int = 48, heads: int = 4, cond_dim: int = 48) -> None:
        """Initialize attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Token feature dimension.
        heads:
            Number of attention heads.
        cond_dim:
            Conditioning vector dimension.
        """
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply a conditioned transformer block.

        Parameters
        ----------
        x:
            Latent video tokens.
        cond:
            Text/timestep conditioning vector.

        Returns
        -------
        Tensor
            Updated tokens.
        """
        h = self.norm1(x, cond)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        return x + self.mlp(self.norm2(x, cond))


class AllegroDiT(nn.Module):
    """Compact Allegro VideoDiT over patchified latent video."""

    def __init__(self, depth: int = 2, dim: int = 48, text_tokens: int = 8) -> None:
        """Initialize patch embedding, text conditioning, and decoder.

        Parameters
        ----------
        depth:
            Number of transformer blocks.
        dim:
            Token feature dimension.
        text_tokens:
            Number of text tokens consumed by the compact text encoder.
        """
        super().__init__()
        self.patch = nn.Conv3d(4, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.text = nn.Embedding(256, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.blocks = nn.ModuleList([AllegroBlock(dim=dim, cond_dim=dim) for _ in range(depth)])
        self.unpatch = nn.ConvTranspose3d(dim, 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.text_tokens = text_tokens

    def forward(self, latent: Tensor, text_ids: Tensor, timestep: Tensor) -> Tensor:
        """Predict denoised latent-video residuals.

        Parameters
        ----------
        latent:
            Latent video tensor ``(batch, 4, frames, height, width)``.
        text_ids:
            Text token ids ``(batch, tokens)``.
        timestep:
            Diffusion timestep scalar tensor ``(batch, 1)``.

        Returns
        -------
        Tensor
            Latent residual with the same shape as ``latent``.
        """
        features = self.patch(latent)
        batch, channels, frames, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        cond = self.text(text_ids[:, : self.text_tokens]).mean(dim=1) + self.time(timestep)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        for block in self.blocks:
            tokens = block(tokens, cond)
        features = tokens.transpose(1, 2).reshape(batch, channels, frames, height, width)
        return self.unpatch(features)


def _build_depth(depth: int) -> nn.Module:
    """Build an Allegro variant with selected transformer depth.

    Parameters
    ----------
    depth:
        Number of transformer blocks.

    Returns
    -------
    nn.Module
        Random-initialized Allegro DiT model.
    """
    return AllegroDiT(depth=depth)


def build() -> nn.Module:
    """Build the default compact Allegro model.

    Returns
    -------
    nn.Module
        Random-initialized compact Allegro model.
    """
    return _build_depth(2)


def build_small() -> nn.Module:
    """Build the smaller Allegro variant.

    Returns
    -------
    nn.Module
        Random-initialized one-block Allegro model.
    """
    return _build_depth(1)


def build_module_r6() -> nn.Module:
    """Build the compact Allegro Module R6 variant.

    Returns
    -------
    nn.Module
        Random-initialized three-block Allegro model.
    """
    return _build_depth(3)


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return latent video, text tokens, and timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Inputs for the compact Allegro model.
    """
    return torch.randn(1, 4, 3, 8, 8), torch.randint(0, 256, (1, 8)), torch.tensor([[0.5]])


MENAGERIE_ENTRIES = [
    ("allegro", "build", "example_input", "2024", "DE"),
    ("Allegro small", "build_small", "example_input", "2024", "DE"),
    ("allegro_Allegro_Module_R6", "build_module_r6", "example_input", "2024", "DE"),
    ("allegro_AllegroModel", "build", "example_input", "2024", "DE"),
    ("allegro_Allegro_oeq", "build_small", "example_input", "2024", "DE"),
]
