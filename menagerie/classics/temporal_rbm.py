"""Temporal RBM, 2007, Sutskever and Hinton.

Paper: Learning Multilevel Distributed Representations for High-Dimensional Sequences.
An RBM with temporal bias links; the recurrent temporal RBM uses a deterministic
hidden state to condition the current undirected RBM.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalRBM(nn.Module):
    """Recurrent temporal RBM unrolled over sequence time."""

    def __init__(self, n_visible: int = 8, n_hidden: int = 6) -> None:
        """Initialize RTRBM parameters.

        Parameters
        ----------
        n_visible:
            Number of visible units per frame.
        n_hidden:
            Number of hidden/recurrent units.
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.visible_hidden = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.05)
        self.recurrent = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.05)
        self.hidden_to_visible = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.03)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.recurrent_bias = nn.Parameter(torch.zeros(n_hidden))

    def step(self, frame: Tensor, recurrent_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one RTRBM conditional step.

        Parameters
        ----------
        frame:
            Current visible frame.
        recurrent_state:
            Previous deterministic recurrent state.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, hidden probabilities, and next recurrent state.
        """
        hidden_bias = self.hidden_bias + recurrent_state @ self.recurrent
        visible_bias = self.visible_bias + recurrent_state @ self.hidden_to_visible
        hidden = torch.sigmoid(frame @ self.visible_hidden + hidden_bias)
        reconstruction = torch.sigmoid(hidden @ self.visible_hidden.T + visible_bias)
        next_state = torch.sigmoid(
            frame @ self.visible_hidden + recurrent_state @ self.recurrent + self.recurrent_bias
        )
        return reconstruction, hidden, next_state

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Unroll recurrent temporal RBM dynamics.

        Parameters
        ----------
        sequence:
            Sequence tensor of shape ``(batch, time, n_visible)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstructions, hidden probabilities, and recurrent states.
        """
        state = torch.zeros(
            sequence.shape[0], self.n_hidden, device=sequence.device, dtype=sequence.dtype
        )
        recons = []
        hiddens = []
        states = []
        for index in range(sequence.shape[1]):
            reconstruction, hidden, state = self.step(sequence[:, index, :], state)
            recons.append(reconstruction)
            hiddens.append(hidden)
            states.append(state)
        return torch.stack(recons, dim=1), torch.stack(hiddens, dim=1), torch.stack(states, dim=1)


def build() -> nn.Module:
    """Build a small RTRBM.

    Returns
    -------
    nn.Module
        TemporalRBM instance.
    """
    return TemporalRBM()


def example_input() -> Tensor:
    """Return a sample sequence batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 5, 8)``.
    """
    return torch.rand(2, 5, 8)
