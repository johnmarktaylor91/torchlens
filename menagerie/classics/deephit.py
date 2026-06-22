"""DeepHit competing-risks survival model.

Paper: A Deep Learning Approach to Survival Analysis with Competing Risks, Lee et al. 2018.

DeepHit directly predicts a discrete joint distribution over event time and
competing-risk cause using a shared covariate trunk plus cause-specific heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepHit(nn.Module):
    """Compact DeepHit distributional survival network."""

    def __init__(
        self, in_features: int = 10, hidden: int = 32, risks: int = 3, times: int = 8
    ) -> None:
        """Initialize shared and cause-specific subnetworks.

        Parameters
        ----------
        in_features:
            Number of covariates.
        hidden:
            Hidden width.
        risks:
            Number of competing causes.
        times:
            Number of discrete event-time bins.
        """

        super().__init__()
        self.risks = risks
        self.times = times
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.risk_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden + in_features, hidden), nn.ReLU(), nn.Linear(hidden, times)
                )
                for _ in range(risks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a normalized joint risk-time distribution.

        Parameters
        ----------
        x:
            Covariates of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(batch, risks, times)``.
        """

        shared = self.shared(x)
        head_in = torch.cat([shared, x], dim=-1)
        logits = torch.stack([head(head_in) for head in self.risk_heads], dim=1)
        return F.softmax(logits.flatten(1), dim=-1).view(x.shape[0], self.risks, self.times)


def build() -> nn.Module:
    """Build compact DeepHit."""

    return DeepHit()


def example_input() -> torch.Tensor:
    """Return tabular survival covariates."""

    return torch.randn(2, 10)


MENAGERIE_ENTRIES = [("DeepHit", "build", "example_input", "2018", "survival/competing-risks")]
