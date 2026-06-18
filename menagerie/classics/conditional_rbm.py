"""Conditional RBM, 2006, Taylor, Hinton, and Roweis.

Paper: Modeling Human Motion Using Binary Latent Variables.
An RBM whose visible and hidden biases are dynamically conditioned on a short
history window; this module also exposes a factored style-gating variant.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionalRBM(nn.Module):
    """History-conditioned RBM for sequence frames."""

    def __init__(
        self, n_visible: int = 8, n_hidden: int = 6, history: int = 2, n_style: int = 3
    ) -> None:
        """Initialize CRBM parameters.

        Parameters
        ----------
        n_visible:
            Number of visible units per frame.
        n_hidden:
            Number of hidden units.
        history:
            Number of previous frames used for dynamic biases.
        n_style:
            Number of optional style dimensions for factored gating.
        """
        super().__init__()
        self.n_visible = n_visible
        self.history = history
        n_cond = history * n_visible
        self.weight = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.05)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.autoregressive = nn.Parameter(torch.randn(n_cond, n_visible) * 0.03)
        self.condition_hidden = nn.Parameter(torch.randn(n_cond, n_hidden) * 0.03)
        self.style_visible = nn.Parameter(torch.randn(n_style, n_visible) * 0.02)
        self.style_hidden = nn.Parameter(torch.randn(n_style, n_hidden) * 0.02)

    def _history_vector(self, sequence: Tensor, index: int) -> Tensor:
        """Flatten the fixed history window before an index.

        Parameters
        ----------
        sequence:
            Input sequence of shape ``(batch, time, n_visible)``.
        index:
            Current sequence index.

        Returns
        -------
        Tensor
            Flattened conditioning vector.
        """
        frames = []
        for offset in range(self.history, 0, -1):
            src = max(index - offset, 0)
            frames.append(sequence[:, src, :])
        return torch.cat(frames, dim=-1)

    def conditional_step(
        self, frame: Tensor, condition: Tensor, style: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute hidden probabilities and reconstruction for one frame.

        Parameters
        ----------
        frame:
            Current visible frame.
        condition:
            Flattened history vector.
        style:
            Optional style vector for factored bias modulation.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstruction and hidden probabilities.
        """
        dyn_visible = self.visible_bias + condition @ self.autoregressive
        dyn_hidden = self.hidden_bias + condition @ self.condition_hidden
        if style is not None:
            dyn_visible = dyn_visible + style @ self.style_visible
            dyn_hidden = dyn_hidden + style @ self.style_hidden
        hidden = torch.sigmoid(frame @ self.weight + dyn_hidden)
        reconstruction = hidden @ self.weight.T + dyn_visible
        return reconstruction, hidden

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Unroll CRBM conditionals over a sequence.

        Parameters
        ----------
        sequence:
            Input sequence of shape ``(batch, time, n_visible)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstructions and hidden probabilities for each step.
        """
        recons = []
        hiddens = []
        for index in range(sequence.shape[1]):
            condition = self._history_vector(sequence, index)
            reconstruction, hidden = self.conditional_step(sequence[:, index, :], condition)
            recons.append(reconstruction)
            hiddens.append(hidden)
        return torch.stack(recons, dim=1), torch.stack(hiddens, dim=1)


def build() -> nn.Module:
    """Build a small CRBM.

    Returns
    -------
    nn.Module
        ConditionalRBM instance.
    """
    return ConditionalRBM()


def example_input() -> Tensor:
    """Return a sample sequence batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 5, 8)``.
    """
    return torch.rand(2, 5, 8)
