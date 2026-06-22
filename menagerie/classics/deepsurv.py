"""DeepSurv Cox proportional-hazards neural network.

Paper: DeepSurv: personalized treatment recommender system using a Cox proportional hazards
deep neural network, Katzman et al. 2018.

DeepSurv replaces the Cox model's linear log-risk with a nonlinear MLP and can
compare counterfactual treatment covariates through the same risk function.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSurv(nn.Module):
    """Compact nonlinear Cox log-risk network."""

    def __init__(self, in_features: int = 12, hidden: int = 32) -> None:
        """Initialize the prognostic risk network.

        Parameters
        ----------
        in_features:
            Patient covariate count including treatment indicator.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SELU(),
            nn.AlphaDropout(0.0),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict patient log-risk.

        Parameters
        ----------
        x:
            Covariates of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Cox log-risk scores.
        """

        return self.net(x)


def build() -> nn.Module:
    """Build compact DeepSurv."""

    return DeepSurv()


def example_input() -> torch.Tensor:
    """Return patient covariates."""

    return torch.randn(3, 12)


MENAGERIE_ENTRIES = [("DeepSurv", "build", "example_input", "2018", "survival/cox")]
