"""Elastic Net, 1987, Durbin and Willshaw, "An Analogue Approach to the Travelling Salesman Problem".

A ring of points is pulled toward cities through Gaussian assignment weights while
an elastic tension term keeps neighboring ring points smooth.
"""

import torch
from torch import Tensor, nn


class ElasticNetTSP(nn.Module):
    """Durbin-Willshaw deformable ring energy module."""

    def __init__(self, n_points: int = 12, k: float = 0.3, beta: float = 0.2) -> None:
        """Initialize ring points and energy constants.

        Parameters
        ----------
        n_points:
            Number of points in the elastic ring.
        k:
            Gaussian assignment width.
        beta:
            Tension penalty strength.
        """
        super().__init__()
        theta = torch.linspace(0.0, 2.0 * torch.pi, n_points + 1)[:-1]
        points = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1) * 0.5
        self.points = nn.Parameter(points)
        self.k = k
        self.beta = beta

    def forward(self, cities: Tensor) -> Tensor:
        """Compute assignment, tension, and total elastic-net energy terms.

        Parameters
        ----------
        cities:
            City coordinates with shape ``(n_cities, 2)``.

        Returns
        -------
        Tensor
            Tensor ``[total_energy, assignment_energy, tension_energy]``.
        """
        diffs = cities[:, None, :] - self.points[None, :, :]
        dist2 = diffs.pow(2).sum(dim=-1)
        assignment = -self.k * torch.logsumexp(-dist2 / (2.0 * self.k * self.k), dim=1).sum()
        tension_vec = (
            torch.roll(self.points, shifts=-1, dims=0)
            - 2.0 * self.points
            + torch.roll(self.points, shifts=1, dims=0)
        )
        tension = self.beta * tension_vec.pow(2).sum()
        return torch.stack((assignment + tension, assignment, tension))


def build() -> nn.Module:
    """Build a small elastic-net TSP module.

    Returns
    -------
    nn.Module
        Configured ``ElasticNetTSP`` instance.
    """
    return ElasticNetTSP()


def example_input() -> Tensor:
    """Create a small city-coordinate example.

    Returns
    -------
    Tensor
        Example input with shape ``(6, 2)``.
    """
    return torch.rand(6, 2) * 2.0 - 1.0
