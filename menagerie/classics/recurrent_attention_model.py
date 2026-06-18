"""Recurrent Attention Model, 2014, Volodymyr Mnih et al.

Paper: Recurrent Models of Visual Attention.
A differentiable retina extracts foveated glimpses, an LSTM core accumulates
evidence, and a location network predicts the next normalized fixation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [("Recurrent Attention Model (RAM)", "build", "example_input", "2014", "DC")]


class RecurrentAttentionModel(nn.Module):
    """Small deterministic RAM classifier with foveated grid-sample glimpses."""

    def __init__(self, glimpse_size: int = 12, hidden_size: int = 32, steps: int = 3) -> None:
        """Initialize glimpse encoder, recurrent core, and heads.

        Parameters
        ----------
        glimpse_size
            Spatial size sampled at each scale.
        hidden_size
            LSTM hidden-state size.
        steps
            Number of recurrent glimpses.
        """
        super().__init__()
        self.glimpse_size = glimpse_size
        self.steps = steps
        self.glimpse_net = nn.Linear(2 * glimpse_size * glimpse_size + 2, hidden_size)
        self.core = nn.LSTMCell(hidden_size, hidden_size)
        self.loc_net = nn.Linear(hidden_size, 2)
        self.classifier = nn.Linear(hidden_size, 10)

    def _glimpse(self, x: Tensor, loc: Tensor, scale: float) -> Tensor:
        """Sample a square glimpse centered at a normalized location.

        Parameters
        ----------
        x
            Image tensor with shape ``(B, 1, H, W)``.
        loc
            Normalized ``xy`` locations in ``[-1, 1]`` with shape ``(B, 2)``.
        scale
            Relative size of the sampled field.

        Returns
        -------
        Tensor
            Sampled glimpse tensor.
        """
        batch = x.shape[0]
        theta = x.new_zeros(batch, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, :, 2] = loc
        grid = F.affine_grid(
            theta, (batch, 1, self.glimpse_size, self.glimpse_size), align_corners=False
        )
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image through sequential deterministic glimpses.

        Parameters
        ----------
        x
            Grayscale image tensor with shape ``(B, 1, 60, 60)``.

        Returns
        -------
        Tensor
            Class logits from the final recurrent state.
        """
        batch = x.shape[0]
        loc = x.new_zeros(batch, 2)
        h = x.new_zeros(batch, self.core.hidden_size)
        c = x.new_zeros(batch, self.core.hidden_size)
        for _ in range(self.steps):
            fine = self._glimpse(x, loc, 0.35).flatten(1)
            coarse = self._glimpse(x, loc, 0.7).flatten(1)
            encoded = torch.relu(self.glimpse_net(torch.cat((fine, coarse, loc), dim=1)))
            h, c = self.core(encoded, (h, c))
            loc = torch.tanh(self.loc_net(h))
        return self.classifier(h)


def build() -> nn.Module:
    """Build a compact recurrent attention model.

    Returns
    -------
    nn.Module
        Random-initialized RAM classifier.
    """
    return RecurrentAttentionModel()


def example_input() -> Tensor:
    """Return a traceable grayscale image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 60, 60)``.
    """
    return torch.randn(1, 1, 60, 60)
