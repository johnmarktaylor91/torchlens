# FAITHFUL REIMPLEMENTATION from arXiv:1907.10719 (no public code) -- A/B codex
"""LayoutVAE with autoregressive CountVAE and BBoxVAE stages."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Small multilayer perceptron used for LayoutVAE conditioning networks."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        in_dim:
            Input feature size.
        hidden_dim:
            Hidden feature size.
        out_dim:
            Output feature size.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.net(x)


class CountVAE(nn.Module):
    """Autoregressive conditional VAE for per-category object counts."""

    def __init__(self, num_labels: int, hidden_dim: int, latent_dim: int) -> None:
        """Initialize CountVAE.

        Parameters
        ----------
        num_labels:
            Number of object categories.
        hidden_dim:
            Hidden feature size.
        latent_dim:
            Latent code size.
        """
        super().__init__()
        self.num_labels = num_labels
        self.label_set_mlp = MLP(num_labels, hidden_dim, hidden_dim)
        self.label_mlp = MLP(num_labels, hidden_dim, hidden_dim)
        self.history_mlp = MLP(num_labels, hidden_dim, hidden_dim)
        self.condition = nn.Linear(hidden_dim * 3, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.rate = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, label_set: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict count rates autoregressively from a label set.

        Parameters
        ----------
        label_set:
            Multi-hot label-set tensor of shape ``(batch, labels)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Count rates, prior means, and prior log variances.
        """
        batch = label_set.shape[0]
        previous_counts = torch.zeros(
            batch, self.num_labels, device=label_set.device, dtype=label_set.dtype
        )
        rates: list[torch.Tensor] = []
        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        context = self.label_set_mlp(label_set)
        for label_index in range(self.num_labels):
            one_hot = torch.zeros_like(label_set)
            one_hot[:, label_index] = 1.0
            cond = torch.cat(
                [context, self.label_mlp(one_hot), self.history_mlp(previous_counts)],
                dim=-1,
            )
            cond = torch.relu(self.condition(cond))
            mu = self.prior_mu(cond)
            logvar = self.prior_logvar(cond)
            latent = mu
            rate = torch.nn.functional.softplus(self.rate(torch.cat([latent, cond], dim=-1))) + 1.0
            rate = rate * label_set[:, label_index : label_index + 1]
            previous_counts[:, label_index : label_index + 1] = rate
            rates.append(rate)
            mus.append(mu)
            logvars.append(logvar)
        return torch.cat(rates, dim=-1), torch.stack(mus, dim=1), torch.stack(logvars, dim=1)


class BBoxVAE(nn.Module):
    """Autoregressive conditional VAE for bounding boxes."""

    def __init__(
        self,
        num_labels: int,
        hidden_dim: int,
        latent_dim: int,
        max_instances: int,
    ) -> None:
        """Initialize BBoxVAE.

        Parameters
        ----------
        num_labels:
            Number of categories.
        hidden_dim:
            Hidden feature size.
        latent_dim:
            Latent code size.
        max_instances:
            Trace-time cap on generated boxes per category.
        """
        super().__init__()
        self.num_labels = num_labels
        self.max_instances = max_instances
        self.count_mlp = MLP(num_labels, hidden_dim, hidden_dim)
        self.label_mlp = MLP(num_labels, hidden_dim, hidden_dim)
        self.history_lstm = nn.LSTM(num_labels + 4, hidden_dim, batch_first=True)
        self.history_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.condition = nn.Linear(hidden_dim * 3, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.box_mean = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(
        self, label_set: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict box means from labels and predicted counts.

        Parameters
        ----------
        label_set:
            Multi-hot tensor of shape ``(batch, labels)``.
        counts:
            Count-rate tensor of shape ``(batch, labels)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Box means and fixed diagonal standard deviations.
        """
        batch = label_set.shape[0]
        count_context = self.count_mlp(counts * label_set)
        previous = torch.zeros(
            batch, 1, self.num_labels + 4, device=label_set.device, dtype=label_set.dtype
        )
        boxes: list[torch.Tensor] = []
        for label_index in range(self.num_labels):
            one_hot = torch.zeros_like(label_set)
            one_hot[:, label_index] = 1.0
            for _ in range(self.max_instances):
                _, (hidden, _) = self.history_lstm(previous)
                hist = self.history_mlp(hidden[-1])
                cond = torch.relu(
                    self.condition(
                        torch.cat([count_context, self.label_mlp(one_hot), hist], dim=-1)
                    )
                )
                mu = self.prior_mu(cond)
                _ = self.prior_logvar(cond)
                box = (
                    self.box_mean(torch.cat([mu, cond], dim=-1))
                    * label_set[:, label_index : label_index + 1]
                )
                boxes.append(box)
                previous_token = torch.cat([one_hot, box], dim=-1).unsqueeze(1)
                previous = torch.cat([previous, previous_token], dim=1)
        means = torch.stack(boxes, dim=1)
        std = torch.full_like(means, 0.02)
        return means, std


class LayoutVAE(nn.Module):
    """Two-stage stochastic scene-layout generator."""

    def __init__(
        self,
        num_labels: int = 5,
        hidden_dim: int = 24,
        latent_dim: int = 8,
        max_instances: int = 2,
    ) -> None:
        """Initialize LayoutVAE.

        Parameters
        ----------
        num_labels:
            Number of categories in the label set universe.
        hidden_dim:
            Hidden dimension.
        latent_dim:
            Latent dimension.
        max_instances:
            Trace-time cap for boxes per label.
        """
        super().__init__()
        self.count_vae = CountVAE(num_labels, hidden_dim, latent_dim)
        self.bbox_vae = BBoxVAE(num_labels, hidden_dim, latent_dim, max_instances)

    def forward(self, label_set: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate count rates and box distributions.

        Parameters
        ----------
        label_set:
            Multi-hot label set tensor of shape ``(batch, labels)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Count rates, box means, and box standard deviations.
        """
        counts, _, _ = self.count_vae(label_set)
        boxes, std = self.bbox_vae(label_set, counts)
        return counts, boxes, std


def build_layoutvae() -> LayoutVAE:
    """Build a tiny traceable LayoutVAE.

    Returns
    -------
    LayoutVAE
        Tiny LayoutVAE instance.
    """
    return LayoutVAE()


def example_input_layoutvae() -> torch.Tensor:
    """Create a label-set input.

    Returns
    -------
    torch.Tensor
        Multi-hot label set.
    """
    return torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [("LayoutVAE", "build_layoutvae", "example_input_layoutvae", 2019, "REIMPL")]
