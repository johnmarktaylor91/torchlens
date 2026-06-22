"""Mimicry InfoMax-GAN generator compact reconstruction.

Paper: Lee and Town, 2021, "InfoMax-GAN: Improved Adversarial Image Generation
via Information Maximization and Contrastive Learning".

Mimicry exposes reproducible GAN baselines; the InfoMax-GAN variant augments
generation with local/global features used by the discriminator's mutual
information and contrastive objectives.  This compact generator preserves the
ResNet upsampling generator and returns multi-scale feature maps for InfoMax
heads rather than being a plain DCGAN.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResGenBlock(nn.Module):
    """Residual upsampling block used by ResNet GAN generators."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize residual and skip projections."""

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Upsample and apply residual generation."""

        up = F.interpolate(x, scale_factor=2, mode="nearest")
        y = F.relu(self.conv1(up))
        y = self.conv2(F.relu(y))
        return y + self.skip(up)


class InfoMaxGANGenerator(nn.Module):
    """Compact ResNet generator exposing local/global InfoMax features."""

    def __init__(self, latent_dim: int = 64, channels: int = 48) -> None:
        """Initialize latent projection and residual upsampling stack."""

        super().__init__()
        self.fc = nn.Linear(latent_dim, channels * 4 * 4)
        self.blocks = nn.ModuleList(
            [
                ResGenBlock(channels, channels),
                ResGenBlock(channels, channels // 2),
                ResGenBlock(channels // 2, channels // 4),
            ]
        )
        self.to_rgb = nn.Conv2d(channels // 4, 3, 3, padding=1)
        self.global_proj = nn.Linear(channels // 4, 32)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Generate an image and InfoMax local/global representations."""

        x = self.fc(z).view(z.shape[0], 48, 4, 4)
        local = x
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 1:
                local = x
        image = torch.tanh(self.to_rgb(F.relu(x)))
        global_code = self.global_proj(x.mean(dim=(2, 3)))
        local_code = F.adaptive_avg_pool2d(local, (4, 4))
        return image, global_code, local_code


def build() -> nn.Module:
    """Build the compact Mimicry InfoMax-GAN generator."""

    return InfoMaxGANGenerator().eval()


def example_input() -> Tensor:
    """Return latent noise for generation."""

    return torch.randn(1, 64)


MENAGERIE_ENTRIES = [
    ("mimicry_infomax_gan_generator", "build", "example_input", "2021", "GEN"),
]
