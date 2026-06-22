"""ConvCNP: Convolutional Conditional Neural Process.

Gordon et al., ICLR 2020.  ConvCNPs embed observed context sets into a
translation-equivariant functional representation using set convolution/density
channels, process that representation with a CNN, and query target locations.
This compact 1D version keeps Gaussian set convolution, density normalization,
CNN processing, and target interpolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactConvCNP(nn.Module):
    """Small one-dimensional ConvCNP regressor."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize ConvCNP modules.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """

        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(0.18))
        self.cnn = nn.Sequential(
            nn.Conv1d(2, channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, 2, 5, padding=2),
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict target function mean and scale.

        Parameters
        ----------
        inputs:
            Tuple ``(context_x, context_y, target_x)``.

        Returns
        -------
        torch.Tensor
            Target mean and positive scale.
        """

        context_x, context_y, target_x = inputs
        grid = torch.linspace(-2.0, 2.0, 32, device=context_x.device).view(1, 1, -1)
        sq = (grid - context_x.unsqueeze(-1)).pow(2)
        weights = torch.exp(-0.5 * sq / self.sigma.square().clamp_min(1e-4))
        density = weights.sum(dim=1)
        signal = (weights * context_y.unsqueeze(-1)).sum(dim=1) / density.clamp_min(1e-4)
        rep = self.cnn(torch.stack([density, signal], dim=1))
        target_weights = torch.softmax(
            -((target_x.unsqueeze(-1) - grid).pow(2)).squeeze(1) / 0.05, dim=-1
        )
        out = torch.matmul(target_weights, rep.transpose(1, 2))
        return torch.stack([out[..., 0], torch.nn.functional.softplus(out[..., 1])], dim=-1)


def build() -> nn.Module:
    """Build compact ConvCNP.

    Returns
    -------
    nn.Module
        Random-init ConvCNP reconstruction.
    """

    return CompactConvCNP()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create context and target points.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Context locations, context values, and target locations.
    """

    cx = torch.linspace(-1.5, 1.5, 8).unsqueeze(0)
    cy = torch.sin(cx * 2.0)
    tx = torch.linspace(-1.8, 1.8, 12).unsqueeze(0)
    return cx, cy, tx


MENAGERIE_ENTRIES = [("ConvCNP", "build", "example_input", "2020", "NP")]
