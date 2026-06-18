"""Deep Kalman Filter, 2015, Krishnan, Shalit, and Sontag.

Paper: "Deep Kalman Filters." A nonlinear gated transition, neural emission
model, and RNN inference network form a sequential VAE with Kalman-like latent
state dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepKalmanFilter(nn.Module):
    """Compact deterministic-forward Deep Kalman Filter."""

    def __init__(self, input_size: int = 32, latent_size: int = 16, hidden_size: int = 48) -> None:
        """Initialize inference, transition, and emission networks.

        Parameters
        ----------
        input_size:
            Observation size.
        latent_size:
            Latent state size.
        hidden_size:
            Inference RNN hidden size.
        """
        super().__init__()
        self.latent_size = latent_size
        self.inference = nn.GRU(input_size, hidden_size, batch_first=True)
        self.to_loc = nn.Linear(hidden_size + latent_size, latent_size)
        self.to_scale = nn.Linear(hidden_size + latent_size, latent_size)
        self.transition_gate = nn.Linear(latent_size, latent_size)
        self.transition_prop = nn.Linear(latent_size, latent_size)
        self.emitter = nn.Sequential(
            nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run deterministic mean-path DKF inference and reconstruction.

        Parameters
        ----------
        x:
            Observation sequence with shape ``(batch, time, input_size)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, latent means, and positive scales.
        """
        rnn_out, _ = self.inference(x)
        latent = x.new_zeros(x.shape[0], self.latent_size)
        latents: list[Tensor] = []
        scales: list[Tensor] = []
        recons: list[Tensor] = []
        for step in range(x.shape[1]):
            trans_gate = torch.sigmoid(self.transition_gate(latent))
            trans_prop = torch.tanh(self.transition_prop(latent))
            prior = trans_gate * trans_prop + (1.0 - trans_gate) * latent
            stats_in = torch.cat((rnn_out[:, step], prior), dim=-1)
            loc = self.to_loc(stats_in)
            scale = torch.nn.functional.softplus(self.to_scale(stats_in)) + 1.0e-4
            latent = loc
            latents.append(latent)
            scales.append(scale)
            recons.append(self.emitter(latent))
        return torch.stack(recons, dim=1), torch.stack(latents, dim=1), torch.stack(scales, dim=1)


def build() -> nn.Module:
    """Build a compact Deep Kalman Filter.

    Returns
    -------
    nn.Module
        Random-initialized DKF.
    """
    return DeepKalmanFilter()


def example_input() -> Tensor:
    """Return an example observation sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 20, 32)``.
    """
    return torch.randn(1, 20, 32)


MENAGERIE_ENTRIES = [("Deep Kalman Filter", "build", "example_input", "2015", "DE")]
