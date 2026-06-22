"""TransGAN compact transformer generator and discriminator.

Jiang et al. (NeurIPS 2021), "TransGAN: Two Pure Transformers Can Make One
Strong GAN".  The generator starts from a latent vector, projects it to a small
token grid, repeatedly applies Transformer blocks, and progressively upsamples
tokens with pixel-shuffle-like resolution increases.  The discriminator is a
ViT-style patch transformer with a class token.

This random-init compact reconstruction keeps those transformer-only inference
paths for CIFAR and CelebA-style generator/discriminator targets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2) -> None:
        """Initialize attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        heads:
            Number of attention heads.
        mlp_ratio:
            Hidden expansion ratio for the feed-forward network.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feed-forward residual updates.

        Parameters
        ----------
        x:
            Token sequence.

        Returns
        -------
        torch.Tensor
            Updated token sequence.
        """

        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class TokenUpsample(nn.Module):
    """TransGAN token upsampling by grid reshape and projection."""

    def __init__(self, dim: int) -> None:
        """Initialize the token projection used after nearest upsampling.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        """

        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, size: int) -> tuple[torch.Tensor, int]:
        """Double the square token grid resolution.

        Parameters
        ----------
        x:
            Token tensor of shape ``(B, H*W, C)``.
        size:
            Current square grid size.

        Returns
        -------
        tuple[torch.Tensor, int]
            Upsampled tokens and new grid size.
        """

        b, _, c = x.shape
        image = x.transpose(1, 2).reshape(b, c, size, size)
        image = F.interpolate(image, scale_factor=2, mode="nearest")
        new_size = size * 2
        tokens = image.flatten(2).transpose(1, 2)
        return self.proj(tokens), new_size


class TransGANGenerator(nn.Module):
    """Compact TransGAN generator."""

    def __init__(self, latent_dim: int = 64, dim: int = 64, out_size: int = 32) -> None:
        """Initialize a progressive transformer generator.

        Parameters
        ----------
        latent_dim:
            Input latent dimension.
        dim:
            Token embedding dimension.
        out_size:
            Output image size.
        """

        super().__init__()
        self.base_size = 4
        self.out_size = out_size
        self.fc = nn.Linear(latent_dim, self.base_size * self.base_size * dim)
        self.pos4 = nn.Parameter(torch.randn(1, self.base_size * self.base_size, dim) * 0.02)
        self.blocks4 = nn.Sequential(TransformerBlock(dim, 4), TransformerBlock(dim, 4))
        self.up1 = TokenUpsample(dim)
        self.blocks8 = nn.Sequential(TransformerBlock(dim, 4))
        self.up2 = TokenUpsample(dim)
        self.blocks16 = nn.Sequential(TransformerBlock(dim, 4))
        self.up3 = TokenUpsample(dim) if out_size >= 32 else nn.Identity()
        self.blocks32 = nn.Sequential(TransformerBlock(dim, 4))
        self.to_rgb = nn.Linear(dim, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate an RGB image from latent vectors.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            RGB image tensor.
        """

        b = z.shape[0]
        size = self.base_size
        x = self.fc(z).view(b, size * size, -1) + self.pos4
        x = self.blocks4(x)
        x, size = self.up1(x, size)
        x = self.blocks8(x)
        x, size = self.up2(x, size)
        x = self.blocks16(x)
        if self.out_size >= 32:
            x, size = self.up3(x, size)  # type: ignore[misc]
            x = self.blocks32(x)
        rgb = torch.tanh(self.to_rgb(x))
        return rgb.transpose(1, 2).reshape(b, 3, size, size)


class TransGANDiscriminator(nn.Module):
    """ViT-style TransGAN discriminator."""

    def __init__(self, image_size: int = 32, patch: int = 4, dim: int = 64) -> None:
        """Initialize patch embedding, class token, and transformer trunk.

        Parameters
        ----------
        image_size:
            Input square image size.
        patch:
            Patch size.
        dim:
            Token embedding dimension.
        """

        super().__init__()
        self.patch = patch
        n = (image_size // patch) ** 2
        self.proj = nn.Linear(3 * patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.Sequential(TransformerBlock(dim, 4), TransformerBlock(dim, 4))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score images as real/fake logits.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Discriminator logits.
        """

        patches = F.unfold(x, kernel_size=self.patch, stride=self.patch).transpose(1, 2)
        tokens = self.proj(patches)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        tokens = self.blocks(tokens)
        return self.head(self.norm(tokens[:, 0]))


def build_generator() -> nn.Module:
    """Build a compact CIFAR TransGAN generator.

    Returns
    -------
    nn.Module
        Generator module.
    """

    return TransGANGenerator(out_size=32)


def build_celeba256_generator() -> nn.Module:
    """Build a compact CelebA-256 TransGAN generator proxy.

    Returns
    -------
    nn.Module
        Generator preserving progressive transformer upsampling.
    """

    return TransGANGenerator(out_size=32, dim=80)


def build_vit_generator_cifar() -> nn.Module:
    """Build the CIFAR ViT generator variant.

    Returns
    -------
    nn.Module
        CIFAR generator.
    """

    return TransGANGenerator(out_size=32, dim=48)


def build_celeba256_discriminator() -> nn.Module:
    """Build the CelebA TransGAN discriminator proxy.

    Returns
    -------
    nn.Module
        ViT discriminator.
    """

    return TransGANDiscriminator(image_size=32, dim=80)


def build_vit_discriminator_cifar() -> nn.Module:
    """Build the CIFAR ViT discriminator.

    Returns
    -------
    nn.Module
        ViT discriminator.
    """

    return TransGANDiscriminator(image_size=32, dim=48)


def example_latent() -> torch.Tensor:
    """Create a latent input.

    Returns
    -------
    torch.Tensor
        Latent tensor of shape ``(1, 64)``.
    """

    return torch.randn(1, 64) / math.sqrt(64)


def example_image() -> torch.Tensor:
    """Create a discriminator image input.

    Returns
    -------
    torch.Tensor
        RGB image of shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "TransGAN generator (pure transformer progressive token upsampler)",
        "build_generator",
        "example_latent",
        "2021",
        "DG",
    ),
    (
        "TransGAN CelebA-256 generator (compact progressive transformer)",
        "build_celeba256_generator",
        "example_latent",
        "2021",
        "DG",
    ),
    (
        "TransGAN ViT generator CIFAR (compact transformer GAN generator)",
        "build_vit_generator_cifar",
        "example_latent",
        "2021",
        "DG",
    ),
    (
        "TransGAN CelebA-256 discriminator (ViT patch transformer)",
        "build_celeba256_discriminator",
        "example_image",
        "2021",
        "DG",
    ),
    (
        "TransGAN ViT discriminator CIFAR (ViT patch transformer)",
        "build_vit_discriminator_cifar",
        "example_image",
        "2021",
        "DG",
    ),
]
