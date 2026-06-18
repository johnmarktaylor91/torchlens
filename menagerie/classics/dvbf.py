"""Deep Variational Bayes Filter, 2017, Karl et al.

Paper: "Deep Variational Bayes Filters." Locally linear latent dynamics mix a
small bank of linear state-space models with neural weights and stochastic-force
inputs; this mean-path version omits sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepVariationalBayesFilter(nn.Module):
    """Compact locally linear DVBF state-space model."""

    def __init__(
        self,
        input_size: int = 16,
        latent_size: int = 12,
        hidden_size: int = 32,
        n_components: int = 4,
    ) -> None:
        """Initialize encoder, mixture dynamics, and emitter.

        Parameters
        ----------
        input_size:
            Observation size.
        latent_size:
            Latent state size.
        hidden_size:
            Hidden size for neural maps.
        n_components:
            Number of local linear dynamics components.
        """
        super().__init__()
        self.latent_size = latent_size
        self.n_components = n_components
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.init_latent = nn.Linear(hidden_size, latent_size)
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_size + input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_components),
        )
        self.force_net = nn.Linear(hidden_size, latent_size)
        self.a_mats = nn.Parameter(torch.randn(n_components, latent_size, latent_size) * 0.05)
        self.b_mats = nn.Parameter(torch.randn(n_components, input_size, latent_size) * 0.05)
        self.c_mats = nn.Parameter(torch.randn(n_components, latent_size, latent_size) * 0.05)
        self.emitter = nn.Sequential(
            nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run locally linear latent filtering and emission.

        Parameters
        ----------
        x:
            Observation sequence with shape ``(batch, time, input_size)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, latent states, and mixture weights.
        """
        encoded, _ = self.encoder(x)
        latent = torch.tanh(self.init_latent(encoded[:, 0]))
        latents: list[Tensor] = []
        alphas: list[Tensor] = []
        recons: list[Tensor] = []
        for step in range(x.shape[1]):
            alpha = torch.softmax(self.alpha_net(torch.cat((latent, x[:, step]), dim=-1)), dim=-1)
            force = torch.tanh(self.force_net(encoded[:, step]))
            a_part = torch.einsum("bm,mij,bj->bi", alpha, self.a_mats, latent)
            b_part = torch.einsum("bm,mij,bi->bj", alpha, self.b_mats, x[:, step])
            c_part = torch.einsum("bm,mij,bj->bi", alpha, self.c_mats, force)
            latent = torch.tanh(a_part + b_part + c_part)
            latents.append(latent)
            alphas.append(alpha)
            recons.append(self.emitter(latent))
        return torch.stack(recons, dim=1), torch.stack(latents, dim=1), torch.stack(alphas, dim=1)


def build() -> nn.Module:
    """Build a compact Deep Variational Bayes Filter.

    Returns
    -------
    nn.Module
        Random-initialized DVBF.
    """
    return DeepVariationalBayesFilter()


def example_input() -> Tensor:
    """Return an example observation sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 20, 16)``.
    """
    return torch.randn(1, 20, 16)


MENAGERIE_ENTRIES = [
    ("Deep Variational Bayes Filter (DVBF)", "build", "example_input", "2017", "DE")
]
