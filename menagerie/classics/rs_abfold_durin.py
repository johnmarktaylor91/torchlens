# FAITHFUL REIMPLEMENTATION from arXiv:2104.04704 (no public code) -- A/B codex
"""DuRIN: deep-unfolded firm-thresholding reflectivity inversion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FirmThreshold(nn.Module):
    """Weighted MCP proximal operator used by DuRIN-1."""

    def __init__(self, length: int) -> None:
        """Create trainable positive thresholds and MCP shape parameters.

        Parameters
        ----------
        length:
            Number of reflectivity samples.
        """
        super().__init__()
        self.raw_mu = nn.Parameter(torch.full((length,), -2.0))
        self.raw_gamma = nn.Parameter(torch.full((length,), 0.7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply componentwise firm thresholding.

        Parameters
        ----------
        x:
            Candidate reflectivity vector.

        Returns
        -------
        torch.Tensor
            Thresholded reflectivity vector.
        """
        mu = F.softplus(self.raw_mu) + 1.0e-4
        gamma = F.softplus(self.raw_gamma) + 1.01
        abs_x = x.abs()
        sign_x = x.sign()
        middle = sign_x * (abs_x - mu) / (1.0 - (1.0 / gamma))
        zeros = torch.zeros_like(x)
        return torch.where(abs_x <= mu, zeros, torch.where(abs_x <= gamma * mu, middle, x))


class DuRINLayer(nn.Module):
    """One unfolded IFTA layer ``x(k+1)=G(Wy + Sx(k))``."""

    def __init__(self, length: int) -> None:
        """Create one DuRIN layer.

        Parameters
        ----------
        length:
            Number of samples in a 1-D seismic trace.
        """
        super().__init__()
        self.measurement = nn.Linear(length, length, bias=False)
        self.recurrent = nn.Linear(length, length, bias=False)
        self.threshold = FirmThreshold(length)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run one unfolded firm-thresholding iteration.

        Parameters
        ----------
        y:
            Observed seismic trace.
        x:
            Current reflectivity estimate.

        Returns
        -------
        torch.Tensor
            Updated reflectivity estimate.
        """
        return self.threshold(self.measurement(y) + self.recurrent(x))


class DuRIN(nn.Module):
    """Deep-unfolded reflectivity inversion network."""

    def __init__(self, length: int = 64, layers: int = 6) -> None:
        """Create the unrolled DuRIN-1 variant.

        Parameters
        ----------
        length:
            Number of samples in a 1-D seismic trace.
        layers:
            Number of unfolded IFTA layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(DuRINLayer(length) for _ in range(layers))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Invert a 1-D seismic trace into sparse reflectivity.

        Parameters
        ----------
        y:
            Tensor of shape ``(batch, 1, length)``.

        Returns
        -------
        torch.Tensor
            Estimated reflectivity with shape ``(batch, 1, length)``.
        """
        y_flat = y.squeeze(1)
        x = torch.zeros_like(y_flat)
        for layer in self.layers:
            x = layer(y_flat, x)
        return x.unsqueeze(1)


def build_durin() -> DuRIN:
    """Build a small DuRIN model.

    Returns
    -------
    DuRIN
        The reimplemented model.
    """
    return DuRIN(length=64, layers=6)


def example_input_durin() -> torch.Tensor:
    """Create an example 1-D seismic trace.

    Returns
    -------
    torch.Tensor
        Example trace of shape ``(1, 1, 64)``.
    """
    return torch.randn(1, 1, 64)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [("DuRIN", "build_durin", "example_input_durin", 2021, "REIMPL")]
