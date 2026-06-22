"""cryoDRGN compact random-init reconstruction.

Paper: cryoDRGN: reconstruction of heterogeneous cryo-EM structures using
neural networks (Zhong et al., Nature Methods 2021).

cryoDRGN is a VAE over particle images whose latent conformation code conditions
an implicit neural representation of 3D density; known pose/orientation projects
coordinates through the volume decoder.  This compact version keeps the image
encoder, reparameterized latent code, pose-conditioned coordinate MLP, and
Fourier-style coordinate features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CryoDRGN(nn.Module):
    """Compact cryoDRGN spatial VAE with neural density decoder."""

    def __init__(self, latent: int = 8, hidden: int = 48, points: int = 16) -> None:
        """Initialize image encoder and pose-conditioned implicit decoder."""

        super().__init__()
        self.points = points
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)
        self.coord = nn.Linear(3 * 3 + latent, hidden)
        self.density = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Decode pose-conditioned 3D density samples from a particle image."""

        image, coords, rotation = inputs
        feat = self.encoder(image)
        mu = self.mu(feat)
        logvar = self.logvar(feat)
        z = mu + torch.exp(0.5 * logvar) * torch.zeros_like(mu)
        rotated = torch.matmul(coords, rotation.transpose(-1, -2))
        fourier = torch.cat([rotated, torch.sin(rotated), torch.cos(rotated)], dim=-1)
        hidden = self.coord(
            torch.cat([fourier, z[:, None, :].expand(-1, coords.shape[1], -1)], dim=-1)
        )
        return self.density(hidden), mu, logvar


def build() -> nn.Module:
    """Build a compact random-init cryoDRGN model."""

    return CryoDRGN().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return a particle image, 3D sample coordinates, and rotation matrix."""

    return (torch.randn(1, 1, 32, 32), torch.randn(1, 16, 3), torch.eye(3).unsqueeze(0))


MENAGERIE_ENTRIES = [
    ("cryoDRGN", "build", "example_input", "2021", "DC"),
]
