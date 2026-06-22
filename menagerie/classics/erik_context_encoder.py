"""Context Encoder compact random-init reconstruction.

Paper: Context Encoders: Feature Learning by Inpainting (Pathak,
Krahenbuhl, Donahue, Darrell, Efros, CVPR 2016).

The distinctive architecture is an encoder-decoder trained to fill missing image
regions, with a channel-wise fully connected bottleneck connecting encoder and
decoder and an adversarial discriminator head for the generated patch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelWiseFullyConnected(nn.Module):
    """Apply the context-encoder channel-wise fully connected bottleneck."""

    def __init__(self, channels: int, spatial: int) -> None:
        """Initialize one spatial mixing matrix shared across channels."""

        super().__init__()
        self.spatial = spatial
        self.weight = nn.Parameter(
            torch.randn(channels, spatial * spatial, spatial * spatial) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(channels, spatial * spatial))

    def forward(self, x: Tensor) -> Tensor:
        """Mix every channel globally over the spatial map."""

        bsz, channels, height, width = x.shape
        flat = x.view(bsz, channels, height * width)
        mixed = torch.einsum("bcn,cmn->bcm", flat, self.weight) + self.bias.unsqueeze(0)
        return mixed.view(bsz, channels, self.spatial, self.spatial)


class ContextEncoder(nn.Module):
    """Compact context encoder for center-region inpainting."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize encoder, channel-wise FC, decoder, and discriminator head."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.channel_fc = ChannelWiseFullyConnected(channels * 2, 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.discriminator = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Generate missing content from a masked image and binary mask."""

        image, mask = inputs
        masked = image * (1.0 - mask)
        encoded = self.encoder(torch.cat([masked, mask], dim=1))
        latent = self.channel_fc(encoded)
        patch = self.decoder(latent)
        return patch, self.discriminator(patch)


def build() -> nn.Module:
    """Build a compact random-init context encoder."""

    return ContextEncoder().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return an RGB image and a center-hole mask."""

    image = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0
    return image, mask


MENAGERIE_ENTRIES = [
    ("erik_context_encoder", "build", "example_input", "2016", "DC"),
]
