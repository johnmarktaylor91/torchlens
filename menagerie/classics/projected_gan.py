"""Projected GAN compact generator, projector, and discriminator components.

Sauer et al., NeurIPS 2021, arXiv:2111.01007.
Paper: Projected GANs Converge Faster.

Projected GAN trains a generator against discriminators that operate on frozen
feature-network projections.  The paper commonly pairs the projected
discriminator with a FastGAN-style upsampling generator.  These compact models
preserve the feature-pyramid projector, multiscale discriminator heads, FastGAN
upsampling blocks, and a small StyleGAN2-like modulated generator variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """FastGAN-style upsample-convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the upsampling block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample and convolve.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Upsampled feature map.
        """

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return F.relu_(self.bn(self.conv(x)))


class FastGANGenerator(nn.Module):
    """Compact FastGAN generator used by Projected GAN."""

    def __init__(self, z_dim: int = 32) -> None:
        """Initialize the generator.

        Parameters
        ----------
        z_dim:
            Latent vector dimension.
        """

        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.blocks = nn.Sequential(
            UpsampleBlock(128, 96),
            UpsampleBlock(96, 64),
            UpsampleBlock(64, 32),
        )
        self.to_rgb = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate an image.

        Parameters
        ----------
        z:
            Latent tensor ``(B, z_dim)``.

        Returns
        -------
        torch.Tensor
            Generated image ``(B, 3, 32, 32)``.
        """

        x = self.fc(z).view(z.shape[0], 128, 4, 4)
        return torch.tanh(self.to_rgb(self.blocks(x)))


class RandomProjector(nn.Module):
    """Frozen random feature pyramid for projected discrimination."""

    def __init__(self) -> None:
        """Initialize frozen projection convolutions."""

        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, 16, 3, padding=1),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.Conv2d(32, 48, 3, stride=2, padding=1),
            ]
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return projected feature maps at multiple scales.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        list[torch.Tensor]
            Feature pyramid.
        """

        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        return features


class ProjectedDiscriminator(nn.Module):
    """Projected GAN multiscale discriminator."""

    def __init__(self) -> None:
        """Initialize projector and discriminator heads."""

        super().__init__()
        self.projector = RandomProjector()
        self.heads = nn.ModuleList([nn.Conv2d(ch, 1, 3, padding=1) for ch in (16, 32, 48)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score an image using projected feature maps.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Scalar discriminator score per batch item.
        """

        scores = []
        for feature, head in zip(self.projector(x), self.heads, strict=True):
            scores.append(head(feature).mean(dim=(1, 2, 3)))
        return torch.stack(scores, dim=-1).sum(dim=-1, keepdim=True)


class ModulatedConv(nn.Module):
    """Small StyleGAN2-like modulated convolution."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int) -> None:
        """Initialize modulation and convolution weights.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        style_dim:
            Style vector dimension.
        """

        super().__init__()
        self.style = nn.Linear(style_dim, in_channels)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply modulated convolution.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Style vector.

        Returns
        -------
        torch.Tensor
            Modulated feature map.
        """

        batch = x.shape[0]
        scale = self.style(style).view(batch, 1, -1, 1, 1) + 1.0
        weight = self.weight.unsqueeze(0) * scale
        x = x.reshape(1, batch * x.shape[1], x.shape[2], x.shape[3])
        weight = weight.reshape(batch * self.weight.shape[0], self.weight.shape[1], 3, 3)
        out = F.conv2d(x, weight, padding=1, groups=batch)
        return out.view(batch, self.weight.shape[0], out.shape[2], out.shape[3]) + self.bias.view(
            1, -1, 1, 1
        )


class StyleGAN2TinyGenerator(nn.Module):
    """Compact StyleGAN2-like generator variant for Projected GAN catalogs."""

    def __init__(self, z_dim: int = 32) -> None:
        """Initialize mapping and modulated synthesis layers.

        Parameters
        ----------
        z_dim:
            Latent vector dimension.
        """

        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(z_dim, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32))
        self.const = nn.Parameter(torch.randn(1, 64, 4, 4))
        self.conv1 = ModulatedConv(64, 48, 32)
        self.conv2 = ModulatedConv(48, 32, 32)
        self.to_rgb = nn.Conv2d(32, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate an image from a latent vector.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Generated image.
        """

        style = self.mapping(z)
        x = self.const.expand(z.shape[0], -1, -1, -1)
        x = F.leaky_relu(self.conv1(x, style), 0.2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.leaky_relu(self.conv2(x, style), 0.2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return torch.tanh(self.to_rgb(x))


def build_fastgan_generator() -> nn.Module:
    """Build the FastGAN-style generator.

    Returns
    -------
    nn.Module
        Random-init generator.
    """

    return FastGANGenerator()


def build_projector() -> nn.Module:
    """Build the random feature projector.

    Returns
    -------
    nn.Module
        Random-init frozen projector.
    """

    return RandomProjector()


def build_discriminator() -> nn.Module:
    """Build the projected discriminator.

    Returns
    -------
    nn.Module
        Random-init projected discriminator.
    """

    return ProjectedDiscriminator()


def build_stylegan2_generator() -> nn.Module:
    """Build the StyleGAN2-like generator.

    Returns
    -------
    nn.Module
        Random-init StyleGAN2-like generator.
    """

    return StyleGAN2TinyGenerator()


def example_latent() -> torch.Tensor:
    """Create a compact latent vector.

    Returns
    -------
    torch.Tensor
        Latent tensor ``(1, 32)``.
    """

    return torch.randn(1, 32)


def example_image() -> torch.Tensor:
    """Create a compact image input.

    Returns
    -------
    torch.Tensor
        Image tensor ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "projected_gan_fastgan_generator",
        "build_fastgan_generator",
        "example_latent",
        "2021",
        "GAN",
    ),
    (
        "projected_gan_stylegan2_generator",
        "build_stylegan2_generator",
        "example_latent",
        "2021",
        "GAN",
    ),
    ("projected_gan_random_projector", "build_projector", "example_image", "2021", "GAN"),
    (
        "projected_gan_projected_discriminator",
        "build_discriminator",
        "example_image",
        "2021",
        "GAN",
    ),
    ("projected_gan_discriminator", "build_discriminator", "example_image", "2021", "GAN"),
]
