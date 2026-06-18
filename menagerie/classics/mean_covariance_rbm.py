"""Mean-Covariance RBM, 2010, Ranzato and Hinton.

Paper: Modeling Pixel Means and Covariances Using Factorized Third-Order RBMs.
Combines Gaussian-Bernoulli mean units with covariance units driven by squared
factor-filter responses; includes a slab mean as in spike-and-slab RBMs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MeanCovarianceRBM(nn.Module):
    """Factored mean-covariance RBM."""

    def __init__(
        self,
        n_visible: int = 16,
        n_factors: int = 12,
        n_covariance: int = 7,
        n_mean: int = 6,
    ) -> None:
        """Initialize mcRBM parameters.

        Parameters
        ----------
        n_visible:
            Number of Gaussian visible units.
        n_factors:
            Number of covariance filter factors.
        n_covariance:
            Number of covariance hidden units.
        n_mean:
            Number of mean hidden units.
        """
        super().__init__()
        self.c_filters = nn.Parameter(torch.randn(n_visible, n_factors) * 0.05)
        self.pooling = nn.Parameter(torch.rand(n_factors, n_covariance) * 0.08)
        self.mean_weight = nn.Parameter(torch.randn(n_visible, n_mean) * 0.05)
        self.covariance_bias = nn.Parameter(torch.zeros(n_covariance))
        self.mean_bias = nn.Parameter(torch.zeros(n_mean))
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.slab_scale = nn.Parameter(torch.ones(n_mean))

    def covariance_prob(self, visible: Tensor) -> Tensor:
        """Compute covariance hidden probabilities.

        Parameters
        ----------
        visible:
            Visible batch.

        Returns
        -------
        Tensor
            Covariance hidden probabilities.
        """
        factor_energy = (visible @ self.c_filters).pow(2)
        logits = -0.5 * (factor_energy @ torch.relu(self.pooling)) + self.covariance_bias
        return torch.sigmoid(logits)

    def mean_prob(self, visible: Tensor) -> Tensor:
        """Compute mean hidden probabilities.

        Parameters
        ----------
        visible:
            Visible batch.

        Returns
        -------
        Tensor
            Mean hidden probabilities.
        """
        return torch.sigmoid(visible @ self.mean_weight + self.mean_bias)

    def free_energy(self, visible: Tensor) -> Tensor:
        """Compute a compact mcRBM free-energy surrogate.

        Parameters
        ----------
        visible:
            Visible batch.

        Returns
        -------
        Tensor
            Per-example free energy.
        """
        centered = visible - self.visible_bias
        gaussian_term = 0.5 * centered.pow(2).sum(dim=-1)
        cov_term = torch.nn.functional.softplus(
            -0.5 * ((visible @ self.c_filters).pow(2) @ torch.relu(self.pooling))
            + self.covariance_bias
        ).sum(dim=-1)
        mean_term = torch.nn.functional.softplus(visible @ self.mean_weight + self.mean_bias).sum(
            dim=-1
        )
        return gaussian_term - cov_term - mean_term

    def forward(self, visible: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute covariance and mean hidden probabilities.

        Parameters
        ----------
        visible:
            Visible batch of shape ``(batch, n_visible)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Covariance probabilities, mean probabilities, slab means, and free energy.
        """
        cov = self.covariance_prob(visible)
        mean = self.mean_prob(visible)
        slab = mean * self.slab_scale
        energy = self.free_energy(visible)
        return cov, mean, slab, energy


def build() -> nn.Module:
    """Build a small mcRBM.

    Returns
    -------
    nn.Module
        MeanCovarianceRBM instance.
    """
    return MeanCovarianceRBM()


def example_input() -> Tensor:
    """Return a sample visible batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 16)``.
    """
    return torch.randn(2, 16) * 0.5
