"""Beta-VAE: disentangling variational autoencoder.

Paper: "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework", Higgins et al., ICLR 2017.

This compact model keeps the standard convolutional VAE encoder/decoder plus
the beta-weighted KL term used to encourage disentangled latent factors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    """Small convolutional beta-VAE."""

    def __init__(self, latent_dim: int = 6, beta: float = 4.0) -> None:
        """Initialize the beta-VAE.

        Parameters
        ----------
        latent_dim:
            Number of latent Gaussian variables.
        beta:
            Multiplier on the KL divergence term.
        """

        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(16 * 4 * 4, latent_dim)
        self.to_logvar = nn.Linear(16 * 4 * 4, latent_dim)
        self.from_z = nn.Linear(latent_dim, 16 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image into diagonal Gaussian parameters.

        Parameters
        ----------
        x:
            Input image batch with shape ``(batch, 1, 16, 16)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance tensors.
        """

        hidden = self.encoder(x)
        return self.to_mu(hidden), self.to_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply a deterministic trace-friendly reparameterization path.

        Parameters
        ----------
        mu:
            Latent Gaussian means.
        logvar:
            Latent Gaussian log variances.

        Returns
        -------
        torch.Tensor
            Latent sample. A zero epsilon keeps atlas generation deterministic.
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.zeros_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variables back to image space.

        Parameters
        ----------
        z:
            Latent tensor with shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed images with shape ``(batch, 1, 16, 16)``.
        """

        hidden = self.from_z(z).view(z.shape[0], 16, 4, 4)
        return self.decoder(hidden)

    def beta_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute the beta-weighted KL regularizer.

        Parameters
        ----------
        mu:
            Latent Gaussian means.
        logvar:
            Latent Gaussian log variances.

        Returns
        -------
        torch.Tensor
            Scalar beta-weighted KL term.
        """

        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return self.beta * kl.mean()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct images and expose beta-VAE latent statistics.

        Parameters
        ----------
        x:
            Input image batch with shape ``(batch, 1, 16, 16)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstruction, latent mean, and scalar beta-weighted KL term.
        """

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, self.beta_kl(mu, logvar)


def build() -> nn.Module:
    """Build a compact beta-VAE.

    Returns
    -------
    nn.Module
        Randomly initialized beta-VAE.
    """

    return BetaVAE()


def example_input() -> torch.Tensor:
    """Create a tiny grayscale image batch.

    Returns
    -------
    torch.Tensor
        Random input with shape ``(1, 1, 16, 16)``.
    """

    return torch.rand(1, 1, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "Beta-VAE (disentangling variational autoencoder)",
        "build",
        "example_input",
        "2017",
        "generative/vae",
    ),
]
