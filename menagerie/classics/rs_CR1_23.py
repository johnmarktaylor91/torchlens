# FAITHFUL REIMPLEMENTATION from https://doi.org/10.1016/j.cnsns.2018.08.028 (no public code)
"""Toy autoencoder EDMD Koopman model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class AEEDMD(nn.Module):
    """Autoencoder with a learned linear latent Koopman operator."""

    def __init__(self, state_dim: int = 5, latent_dim: int = 4) -> None:
        """Initialize the AE-EDMD model.

        Parameters
        ----------
        state_dim:
            State dimension.
        latent_dim:
            Latent observable dimension.
        """
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, latent_dim))
        self.koopman = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.Tanh(), nn.Linear(16, state_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode state, advance latent linearly, and decode next state.

        Parameters
        ----------
        x:
            Dynamical-system state.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, latent next state, and decoded next state.
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        z_next = self.koopman(z)
        x_next = self.decoder(z_next)
        return reconstruction, z_next, x_next


def build_ae_edmd() -> AEEDMD:
    """Build a tiny AE-EDMD model.

    Returns
    -------
    AEEDMD
        Model instance.
    """
    return AEEDMD()


def example_input_ae_edmd() -> Tensor:
    """Create example dynamical-system state.

    Returns
    -------
    Tensor
        State tensor.
    """
    return torch.randn(3, 5)


MENAGERIE_ENTRIES = [("AE-EDMD", build_ae_edmd, example_input_ae_edmd, 2018, "REIMPL")]
