"""Fields of Experts, 2005, Roth and Black.

Paper: Fields of Experts: A Framework for Learning Image Priors.
Product-of-experts image prior whose expert potentials are Student-t-like
functions of learned filter responses over an image field.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FieldsOfExperts(nn.Module):
    """Product of filter experts for image-prior energy."""

    def __init__(self, in_channels: int = 1, n_experts: int = 5, kernel_size: int = 3) -> None:
        """Initialize FoE filters and potential weights.

        Parameters
        ----------
        in_channels:
            Number of image channels.
        n_experts:
            Number of learned filter experts.
        kernel_size:
            Spatial filter size.
        """
        super().__init__()
        self.filters = nn.Parameter(
            torch.randn(n_experts, in_channels, kernel_size, kernel_size) * 0.08
        )
        self.log_alpha = nn.Parameter(torch.zeros(n_experts))

    def potential_map(self, image: Tensor) -> Tensor:
        """Compute per-expert Student-t potential maps.

        Parameters
        ----------
        image:
            Image batch.

        Returns
        -------
        Tensor
            Potential maps of shape ``(batch, n_experts, height, width)`` up to valid padding.
        """
        responses = F.conv2d(image, self.filters)
        alpha = torch.nn.functional.softplus(self.log_alpha).view(1, -1, 1, 1)
        return alpha * torch.log1p(0.5 * responses.pow(2))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Compute product-of-experts energy.

        Parameters
        ----------
        image:
            Image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-example energy and per-expert potential maps.
        """
        potentials = self.potential_map(image)
        energy = potentials.sum(dim=(1, 2, 3))
        return energy, potentials


def build() -> nn.Module:
    """Build a small Fields-of-Experts model.

    Returns
    -------
    nn.Module
        FieldsOfExperts instance.
    """
    return FieldsOfExperts()


def example_input() -> Tensor:
    """Return a sample image batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 1, 16, 16)``.
    """
    return torch.randn(2, 1, 16, 16) * 0.3
