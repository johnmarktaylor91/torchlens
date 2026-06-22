"""UNIT VAE generator compact reconstruction.

Paper: Liu, Breuel, and Kautz, 2017, "Unsupervised Image-to-Image Translation
Networks".

UNIT couples two image domains with shared latent content and a variational
autoencoder/GAN objective.  This compact generator keeps the distinctive
primitive: a domain-specific decoder fed by a reparameterized content code,
AdaIN-style style modulation, residual decoding, and progressive upsampling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AdaINResidualBlock(nn.Module):
    """Residual decoder block with adaptive instance normalization."""

    def __init__(self, channels: int, style_dim: int) -> None:
        """Initialize convolution and style affine layers."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.style = nn.Linear(style_dim, 4 * channels)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """Apply AdaIN residual decoding.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Domain/style vector.

        Returns
        -------
        Tensor
            Updated feature map.
        """

        gamma1, beta1, gamma2, beta2 = self.style(style).chunk(4, dim=-1)
        y = F.instance_norm(self.conv1(x))
        y = F.relu(y * (1 + gamma1[..., None, None]) + beta1[..., None, None])
        y = F.instance_norm(self.conv2(y))
        y = y * (1 + gamma2[..., None, None]) + beta2[..., None, None]
        return F.relu(x + y)


class UNITVAEGenerator(nn.Module):
    """Compact UNIT-style VAE generator for one image domain."""

    def __init__(self, latent_dim: int = 32, style_dim: int = 12, channels: int = 32) -> None:
        """Initialize latent projection, residual decoder, and image head."""

        super().__init__()
        self.mu = nn.Linear(latent_dim, channels * 4 * 4)
        self.logvar = nn.Linear(latent_dim, channels * 4 * 4)
        self.style_embed = nn.Embedding(2, style_dim)
        self.res1 = AdaINResidualBlock(channels, style_dim)
        self.res2 = AdaINResidualBlock(channels, style_dim)
        self.up1 = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.up2 = nn.Conv2d(channels // 2, channels // 4, 3, padding=1)
        self.to_rgb = nn.Conv2d(channels // 4, 3, 3, padding=1)

    def forward(self, z: Tensor, domain: Tensor) -> Tensor:
        """Decode a variational latent code into a domain-conditioned image."""

        mean = self.mu(z)
        logvar = self.logvar(z)
        eps = torch.tanh(mean)
        x = mean + torch.exp(0.5 * logvar) * eps
        x = x.view(z.shape[0], 32, 4, 4)
        style = self.style_embed(domain)
        x = self.res1(x, style)
        x = self.res2(x, style)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.up2(x))
        return torch.tanh(self.to_rgb(x))


def build() -> nn.Module:
    """Build the compact UNIT VAE generator."""

    return UNITVAEGenerator().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return latent vector and target-domain id."""

    return torch.randn(1, 32), torch.tensor([1], dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("unit_vae_generator", "build", "example_input", "2017", "GEN"),
]
