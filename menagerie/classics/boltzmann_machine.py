"""Full Boltzmann Machine, 1985, Ackley, Hinton, and Sejnowski.

Paper: "A Learning Algorithm for Boltzmann Machines." The model is an
unrestricted binary energy model over visible and hidden units with symmetric
lateral connections; forward runs deterministic Gibbs probabilities.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FullBoltzmannMachine(nn.Module):
    """Unrestricted Boltzmann machine with one symmetric zero-diagonal weight."""

    def __init__(self, n_visible: int = 6, n_hidden: int = 4, steps: int = 4) -> None:
        """Initialize the machine.

        Parameters
        ----------
        n_visible:
            Number of clamped visible units.
        n_hidden:
            Number of hidden units.
        steps:
            Number of deterministic Gibbs sweeps used in ``forward``.
        """
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.steps = steps
        n_total = n_visible + n_hidden
        self.weight_raw = nn.Parameter(0.15 * torch.randn(n_total, n_total))
        self.bias = nn.Parameter(torch.zeros(n_total))

    def symmetric_weight(self) -> Tensor:
        """Return the symmetric zero-diagonal Boltzmann weight matrix.

        Returns
        -------
        Tensor
            Symmetric matrix over visible and hidden units.
        """
        weight = 0.5 * (self.weight_raw + self.weight_raw.transpose(0, 1))
        eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        return weight * (1.0 - eye)

    def energy(self, state: Tensor) -> Tensor:
        """Compute the Boltzmann energy for a binary state.

        Parameters
        ----------
        state:
            Tensor with shape ``(B, n_visible + n_hidden)``.

        Returns
        -------
        Tensor
            Per-example energies.
        """
        weight = self.symmetric_weight()
        quadratic = (state @ weight * state).sum(dim=-1)
        linear = (state * self.bias).sum(dim=-1)
        return -0.5 * quadratic - linear

    def forward(self, visible: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run clamped deterministic Gibbs mean updates.

        Parameters
        ----------
        visible:
            Binary visible state with shape ``(B, n_visible)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstructed visible probabilities, hidden probabilities, and
            final full-state energy.
        """
        weight = self.symmetric_weight()
        hidden = torch.full(
            (visible.shape[0], self.n_hidden),
            0.5,
            device=visible.device,
            dtype=visible.dtype,
        )
        state = torch.cat((visible, hidden), dim=-1)
        for _ in range(self.steps):
            logits = state @ weight + self.bias
            probs = torch.sigmoid(logits)
            state = torch.cat((visible, probs[:, self.n_visible :]), dim=-1)
        free_logits = state @ weight + self.bias
        free_probs = torch.sigmoid(free_logits)
        return free_probs[:, : self.n_visible], state[:, self.n_visible :], self.energy(state)


def build() -> nn.Module:
    """Build a small random-init full Boltzmann machine.

    Returns
    -------
    nn.Module
        A traceable ``FullBoltzmannMachine`` instance.
    """
    return FullBoltzmannMachine()


def example_input() -> Tensor:
    """Return a binary visible example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 6)``.
    """
    return torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
