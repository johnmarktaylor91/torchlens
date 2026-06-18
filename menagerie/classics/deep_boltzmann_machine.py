"""Deep Boltzmann Machine, 2009, Salakhutdinov and Hinton.

Paper: Deep Boltzmann Machines.
Fully undirected stacked Boltzmann machine with bottom-up and top-down
mean-field inference between adjacent hidden layers.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepBoltzmannMachine(nn.Module):
    """Small two-hidden-layer Deep Boltzmann Machine."""

    def __init__(self, n_visible: int = 16, n_hidden1: int = 10, n_hidden2: int = 6) -> None:
        """Initialize DBM parameters.

        Parameters
        ----------
        n_visible:
            Number of visible Bernoulli units.
        n_hidden1:
            Number of first-layer hidden Bernoulli units.
        n_hidden2:
            Number of second-layer hidden Bernoulli units.
        """
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_visible, n_hidden1) * 0.05)
        self.w2 = nn.Parameter(torch.randn(n_hidden1, n_hidden2) * 0.05)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden1_bias = nn.Parameter(torch.zeros(n_hidden1))
        self.hidden2_bias = nn.Parameter(torch.zeros(n_hidden2))

    def mean_field_infer(self, visible: Tensor, steps: int = 3) -> tuple[Tensor, Tensor]:
        """Infer hidden mean-field probabilities.

        Parameters
        ----------
        visible:
            Visible batch of shape ``(batch, n_visible)``.
        steps:
            Number of alternating mean-field updates.

        Returns
        -------
        tuple[Tensor, Tensor]
            First and second hidden-layer probabilities.
        """
        q1 = torch.sigmoid(visible @ self.w1 + self.hidden1_bias)
        q2 = torch.sigmoid(q1 @ self.w2 + self.hidden2_bias)
        for _ in range(steps):
            q1 = torch.sigmoid(visible @ self.w1 + q2 @ self.w2.T + self.hidden1_bias)
            q2 = torch.sigmoid(q1 @ self.w2 + self.hidden2_bias)
        return q1, q2

    def reconstruct(self, q1: Tensor) -> Tensor:
        """Reconstruct visible probabilities from first hidden means.

        Parameters
        ----------
        q1:
            First hidden-layer probabilities.

        Returns
        -------
        Tensor
            Visible reconstruction probabilities.
        """
        return torch.sigmoid(q1 @ self.w1.T + self.visible_bias)

    def gibbs_sample(self, visible: Tensor, steps: int = 2) -> tuple[Tensor, Tensor, Tensor]:
        """Run deterministic block-Gibbs probabilities for tracing.

        Parameters
        ----------
        visible:
            Initial visible units.
        steps:
            Number of block updates.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Visible, first hidden, and second hidden probabilities.
        """
        v_prob = visible
        q1, q2 = self.mean_field_infer(v_prob, steps=1)
        for _ in range(steps):
            q1 = torch.sigmoid(v_prob @ self.w1 + q2 @ self.w2.T + self.hidden1_bias)
            q2 = torch.sigmoid(q1 @ self.w2 + self.hidden2_bias)
            v_prob = self.reconstruct(q1)
        return v_prob, q1, q2

    def forward(self, visible: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mean-field hidden probabilities and reconstruction.

        Parameters
        ----------
        visible:
            Visible batch of shape ``(batch, n_visible)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Visible reconstruction, first hidden probabilities, second hidden probabilities.
        """
        q1, q2 = self.mean_field_infer(visible)
        reconstruction = self.reconstruct(q1)
        return reconstruction, q1, q2


def build() -> nn.Module:
    """Build a small random-init DBM.

    Returns
    -------
    nn.Module
        Deep Boltzmann Machine instance.
    """
    return DeepBoltzmannMachine()


def example_input() -> Tensor:
    """Return a sample visible batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 16)``.
    """
    return torch.rand(2, 16)
