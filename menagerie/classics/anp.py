"""ANP: Attentive Neural Process for conditional function regression.

Attentive Neural Processes add self-attention over context points and
cross-attention from target locations to context representations, producing
query-specific predictive distributions instead of a single pooled context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentiveNeuralProcess(nn.Module):
    """Compact attentive neural process."""

    def __init__(self, x_dim: int = 1, y_dim: int = 1, hidden: int = 48) -> None:
        """Initialize encoders, attention layers, and Gaussian head.

        Parameters
        ----------
        x_dim:
            Input coordinate dimension.
        y_dim:
            Observed value dimension.
        hidden:
            Representation dimension.
        """
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.target_encoder = nn.Linear(x_dim, hidden)
        self.self_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * y_dim),
        )

    def forward(
        self, context_x: Tensor, context_y: Tensor, target_x: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict target means and scales from context observations.

        Parameters
        ----------
        context_x:
            Context coordinates ``(batch, context, x_dim)``.
        context_y:
            Context values ``(batch, context, y_dim)``.
        target_x:
            Target coordinates ``(batch, target, x_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Predictive means and positive scales.
        """
        context = self.context_encoder(torch.cat([context_x, context_y], dim=-1))
        context, _ = self.self_attn(context, context, context, need_weights=False)
        query = self.target_encoder(target_x)
        attended, _ = self.cross_attn(query, context, context, need_weights=False)
        stats = self.decoder(torch.cat([attended, target_x], dim=-1))
        mean, raw_scale = stats.chunk(2, dim=-1)
        return mean, 0.05 + torch.nn.functional.softplus(raw_scale)


def build() -> nn.Module:
    """Build a compact Attentive Neural Process.

    Returns
    -------
    nn.Module
        Random-initialized ANP model.
    """
    return AttentiveNeuralProcess()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return context and target regression points.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Context x, context y, and target x tensors.
    """
    context_x = torch.linspace(-1.0, 1.0, 8).view(1, 8, 1)
    context_y = torch.sin(3.14 * context_x)
    target_x = torch.linspace(-1.0, 1.0, 12).view(1, 12, 1)
    return context_x, context_y, target_x


MENAGERIE_ENTRIES = [
    ("ANP", "build", "example_input", "2018", "E6"),
]
