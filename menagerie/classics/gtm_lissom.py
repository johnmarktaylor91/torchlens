"""GTM and LISSOM maps, 1996-1998, Bishop and Miikkulainen.

Paper: Bishop 1998, "GTM: The generative topographic mapping"; Miikkulainen 1996,
"LISSOM." GTM is represented as a differentiable latent-grid mixture posterior, while
LISSOM settles lateral excitatory and inhibitory activity without Hebbian training updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GenerativeTopographicMapping(nn.Module):
    """Probabilistic latent grid with RBF basis and Gaussian responsibilities."""

    def __init__(self, data_dim: int = 12, grid_size: int = 5, n_basis: int = 9) -> None:
        """Initialize latent grid, RBF centers, and data-space map.

        Parameters
        ----------
        data_dim
            Observed data dimensionality.
        grid_size
            Number of points per side in the latent grid.
        n_basis
            Number of RBF basis functions.
        """
        super().__init__()
        coords = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        centers = torch.randn(n_basis, 2) * 0.7
        self.register_buffer("grid", grid)
        self.register_buffer("centers", centers)
        self.weight = nn.Parameter(torch.randn(n_basis, data_dim) * 0.2)
        self.beta = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: Tensor) -> Tensor:
        """Compute GTM latent responsibilities for observations.

        Parameters
        ----------
        x
            Observation tensor with shape ``(batch, 12)``.

        Returns
        -------
        Tensor
            Posterior responsibility over latent grid points.
        """
        dist = torch.cdist(self.grid, self.centers).pow(2)
        basis = torch.exp(-2.0 * dist)
        means = basis @ self.weight
        sq_error = torch.cdist(x, means).pow(2)
        return torch.softmax(-F.softplus(self.beta) * sq_error, dim=-1)


class LISSOM(nn.Module):
    """Lateral self-organizing map settling dynamics."""

    def __init__(self, input_dim: int = 24 * 24, map_size: int = 8, steps: int = 4) -> None:
        """Initialize afferent and lateral weights.

        Parameters
        ----------
        input_dim
            Flattened sensory input dimensionality.
        map_size
            Side length of the cortical activity map.
        steps
            Number of activity settling iterations.
        """
        super().__init__()
        n_units = map_size * map_size
        self.map_size = map_size
        self.steps = steps
        self.afferent = nn.Linear(input_dim, n_units)
        self.lateral_exc = nn.Parameter(torch.randn(n_units, n_units) * 0.03)
        self.lateral_inh = nn.Parameter(torch.randn(n_units, n_units) * 0.03)

    def forward(self, image: Tensor) -> Tensor:
        """Settle lateral map activity for an image stimulus.

        Parameters
        ----------
        image
            Input image tensor with shape ``(batch, 1, 24, 24)``.

        Returns
        -------
        Tensor
            Settled activity map with shape ``(batch, 1, 8, 8)``.
        """
        drive = self.afferent(image.flatten(1))
        activity = torch.sigmoid(drive)
        exc = F.softplus(self.lateral_exc)
        inh = F.softplus(self.lateral_inh)
        for _ in range(self.steps):
            lateral = activity @ exc - activity @ inh
            activity = torch.sigmoid(drive + 0.15 * lateral)
        return activity.reshape(image.shape[0], 1, self.map_size, self.map_size)


MENAGERIE_ENTRIES = [
    (
        "Generative Topographic Mapping (GTM)",
        "build_gtm",
        "example_input_gtm",
        "1998",
        "DA",
    ),
    ("LISSOM", "build_lissom", "example_input_lissom", "1996", "DA"),
]


def build_gtm() -> nn.Module:
    """Build a GTM module.

    Returns
    -------
    nn.Module
        Configured GTM module.
    """
    return GenerativeTopographicMapping()


def example_input_gtm() -> Tensor:
    """Create a GTM observation example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 12)``.
    """
    return torch.randn(1, 12)


def build_lissom() -> nn.Module:
    """Build a LISSOM module.

    Returns
    -------
    nn.Module
        Configured LISSOM module.
    """
    return LISSOM()


def example_input_lissom() -> Tensor:
    """Create a LISSOM image example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 24, 24)``.
    """
    return torch.randn(1, 1, 24, 24)
