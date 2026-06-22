"""BNP compact Bayesian Neural Process.

Paper: Volpp et al., 2020, "Bayesian Context Aggregation for Neural Processes";
Tailor et al., 2023, "Exploiting Inferential Structure in Neural Processes".

Bayesian Neural Processes replace plain mean aggregation with Bayesian
aggregation over context statistics.  This compact model keeps the distinctive
context encoder, precision-weighted Bayesian aggregation, latent sample, and
target decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BayesianAggregator(nn.Module):
    """Precision-weighted aggregator for neural-process context points."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize context-statistic projections."""

        super().__init__()
        self.to_mean = nn.Linear(dim, dim)
        self.to_logprec = nn.Linear(dim, dim)

    def forward(self, reps: Tensor) -> tuple[Tensor, Tensor]:
        """Aggregate context representations as a Gaussian posterior."""

        mean_i = self.to_mean(reps)
        prec_i = F.softplus(self.to_logprec(reps)) + 1e-3
        precision = prec_i.sum(dim=1)
        mean = (mean_i * prec_i).sum(dim=1) / precision
        scale = torch.rsqrt(precision)
        return mean, scale


class BayesianNeuralProcess(nn.Module):
    """Compact BNP regression model."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize encoders and target decoder."""

        super().__init__()
        self.context = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.agg = BayesianAggregator(dim)
        self.target = nn.Sequential(nn.Linear(1 + dim, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(
        self, context_x: Tensor, context_y: Tensor, target_x: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict target distribution from context set."""

        reps = self.context(torch.cat([context_x, context_y], dim=-1))
        mean, scale = self.agg(reps)
        latent = mean + scale * torch.tanh(mean)
        target_latent = latent.unsqueeze(1).expand(-1, target_x.shape[1], -1)
        stats = self.target(torch.cat([target_x, target_latent], dim=-1))
        pred_mean, pred_logstd = stats.chunk(2, dim=-1)
        return pred_mean, F.softplus(pred_logstd)


def build() -> nn.Module:
    """Build the compact Bayesian Neural Process."""

    return BayesianNeuralProcess().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return context x/y values and target x values."""

    return torch.randn(1, 6, 1), torch.randn(1, 6, 1), torch.randn(1, 8, 1)


MENAGERIE_ENTRIES = [
    ("BNP", "build", "example_input", "2020", "PROB"),
]
