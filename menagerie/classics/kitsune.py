"""Kitsune / KitNET: online ensemble autoencoder NIDS.

Paper: "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion
Detection", Mirsky et al., NDSS 2018.

The compact reconstruction keeps KitNET's feature-mapped ensemble of small
autoencoders and the output autoencoder over per-cluster reconstruction errors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyAutoEncoder(nn.Module):
    """Small autoencoder for one feature cluster."""

    def __init__(self, dim: int) -> None:
        """Initialize encoder and decoder."""

        super().__init__()
        hidden = max(2, dim // 2)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct clustered features."""

        return self.net(x)


class KitsuneCompact(nn.Module):
    """Compact KitNET anomaly scorer."""

    def __init__(self, n_features: int = 20) -> None:
        """Initialize feature clusters and output autoencoder."""

        super().__init__()
        self.clusters = [(0, 5), (5, 10), (10, 15), (15, n_features)]
        self.ensemble = nn.ModuleList(
            [TinyAutoEncoder(end - start) for start, end in self.clusters]
        )
        self.output_ae = TinyAutoEncoder(len(self.clusters))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return output-layer reconstruction error anomaly score."""

        errors = []
        for (start, end), ae in zip(self.clusters, self.ensemble, strict=True):
            part = features[:, start:end]
            recon = ae(part)
            errors.append((part - recon).pow(2).mean(dim=1, keepdim=True))
        ensemble_errors = torch.cat(errors, dim=1)
        recon_errors = self.output_ae(ensemble_errors)
        return (ensemble_errors - recon_errors).pow(2).mean(dim=1, keepdim=True)


def build() -> nn.Module:
    """Build compact Kitsune."""

    return KitsuneCompact()


def example_input() -> torch.Tensor:
    """Return network-flow feature vector."""

    return torch.randn(1, 20)


MENAGERIE_ENTRIES = [("Kitsune", "build", "example_input", "2018", "E7")]
