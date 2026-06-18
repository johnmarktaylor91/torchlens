"""Atick-Redlich retinal efficient coding, 1990, Atick and Redlich.

Paper: "Towards a theory of early visual processing." Center-surround filtering and
contrast normalization approximate retinal whitening under signal and noise
constraints.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Atick-Redlich Retinal Efficient Coding", "build", "example_input", "1990", "DA")
]


class AtickRedlichRetina(nn.Module):
    """Difference-of-Gaussians retina with contrast normalization."""

    def __init__(self, kernel_size: int = 9) -> None:
        """Initialize fixed center-surround filters.

        Parameters
        ----------
        kernel_size
            Width of the square retinal filter kernel.
        """
        super().__init__()
        grid = torch.linspace(-2.0, 2.0, kernel_size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        radius2 = xx.pow(2) + yy.pow(2)
        center = torch.exp(-radius2 / 0.5)
        surround = torch.exp(-radius2 / 2.0)
        dog = center / center.sum() - surround / surround.sum()
        lowpass = surround / surround.sum()
        self.register_buffer("dog", dog.view(1, 1, kernel_size, kernel_size))
        self.register_buffer("lowpass", lowpass.view(1, 1, kernel_size, kernel_size))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply center-surround and contrast-normalized retinal coding.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Difference-of-Gaussians response and normalized response.
        """
        padding = self.dog.shape[-1] // 2
        response = F.conv2d(x, self.dog, padding=padding)
        luminance = torch.relu(F.conv2d(x.abs(), self.lowpass, padding=padding))
        normalized = response / torch.sqrt(luminance + 1.0e-4)
        return response, normalized


def build() -> nn.Module:
    """Build a small Atick-Redlich retinal module.

    Returns
    -------
    nn.Module
        Configured ``AtickRedlichRetina`` instance.
    """
    return AtickRedlichRetina()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)
