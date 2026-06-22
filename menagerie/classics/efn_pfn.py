"""Energy Flow Networks and Particle Flow Networks.

Paper: "Energy Flow Networks: Deep Sets for Particle Jets", Komiske,
Metodiev, and Thaler, JHEP 2019.

EFN/PFN are Deep Sets for collider jets.  EFN enforces IRC safety by summing
energy-weighted per-particle angular embeddings, while PFN sums learned
per-particle embeddings over the full feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small fully connected network."""

    def __init__(self, dims: list[int]) -> None:
        """Initialize linear layers.

        Parameters
        ----------
        dims:
            Layer dimensions.
        """

        super().__init__()
        layers: list[nn.Module] = []
        for idx, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(din, dout))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""

        return self.net(x)


class EFNPFNCompact(nn.Module):
    """Joint compact EFN/PFN reconstruction."""

    def __init__(self, particle_dim: int = 3, latent_dim: int = 16) -> None:
        """Initialize EFN and PFN branches."""

        super().__init__()
        self.efn_phi = MLP([2, latent_dim, latent_dim])
        self.efn_f = MLP([latent_dim, latent_dim, 2])
        self.pfn_phi = MLP([particle_dim, latent_dim, latent_dim])
        self.pfn_f = MLP([latent_dim, latent_dim, 2])

    def forward(self, particles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify jets with EFN and PFN summaries.

        Parameters
        ----------
        particles:
            Tensor ``(batch, particles, [z, y, phi])``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            EFN logits and PFN logits.
        """

        z = torch.softmax(particles[..., :1], dim=1)
        angles = particles[..., 1:3]
        efn_summary = (z * self.efn_phi(angles)).sum(dim=1)
        pfn_summary = self.pfn_phi(particles).sum(dim=1)
        return self.efn_f(efn_summary), self.pfn_f(pfn_summary)


def build() -> nn.Module:
    """Build compact EFN/PFN."""

    return EFNPFNCompact()


def example_input() -> torch.Tensor:
    """Return a small particle jet cloud."""

    return torch.randn(1, 16, 3)


MENAGERIE_ENTRIES = [("EFN/PFN", "build", "example_input", "2019", "E7")]
