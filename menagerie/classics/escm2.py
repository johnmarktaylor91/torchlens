"""ESCM2: Entire Space Counterfactual Multi-Task Model for post-click CVR.

Paper: Wang et al. 2022, "ESCM2: Entire Space Counterfactual Multi-Task Model
for Post-Click Conversion Rate Estimation" (SIGIR), arXiv:2204.05125.

The compact reconstruction keeps the defining ESMM-style shared embedding with
three towers: empirical CTR, global CTCVR, counterfactual CVR, plus the
ESCM2-DR imputation-error tower used to correct inverse-propensity bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """Small MLP tower used by the ESCM2 task heads."""

    def __init__(self, dim: int) -> None:
        """Initialize a two-layer task tower.

        Parameters
        ----------
        dim:
            Input and hidden feature width.
        """

        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project shared features to one task logit.

        Parameters
        ----------
        x:
            Shared dense feature tensor.

        Returns
        -------
        torch.Tensor
            Task logit tensor.
        """

        return self.net(x)


class ESCM2(nn.Module):
    """Counterfactual entire-space multi-task recommender."""

    def __init__(self, vocab: int = 32, fields: int = 6, dim: int = 16) -> None:
        """Initialize embeddings and ESCM2 task towers.

        Parameters
        ----------
        vocab:
            Feature vocabulary size.
        fields:
            Number of categorical fields.
        dim:
            Embedding and tower width.
        """

        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.shared = nn.Sequential(nn.Linear(fields * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ctr = Tower(dim)
        self.cvr = Tower(dim)
        self.ctcvr = Tower(dim)
        self.impute = Tower(dim)
        self.propensity = Tower(dim)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictions plus IPS/DR counterfactual risk terms.

        Parameters
        ----------
        ids:
            Integer feature ids with shape ``(batch, fields)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predictions and differentiable ESCM2 IPS/DR risk terms.
        """

        feat = self.embedding(ids).flatten(1)
        shared = F.relu(self.shared(feat))
        ctr = torch.sigmoid(self.ctr(shared))
        cvr_cf = torch.sigmoid(self.cvr(shared))
        ctcvr_direct = torch.sigmoid(self.ctcvr(shared))
        esmm_ctcvr = ctr * cvr_cf
        imputed = torch.sigmoid(self.impute(shared))
        propensity = torch.sigmoid(self.propensity(shared)).clamp(0.05, 1.0)
        click_proxy = (ids[:, :1].float() % 2.0).detach()
        conversion_proxy = (ids[:, 1:2].float() % 2.0).detach()
        ips_cvr_risk = (
            click_proxy
            * F.binary_cross_entropy(cvr_cf, conversion_proxy, reduction="none")
            / propensity
        )
        imputation_error = F.binary_cross_entropy(imputed, conversion_proxy, reduction="none")
        dr_cvr_risk = (
            imputation_error
            + click_proxy
            * (
                F.binary_cross_entropy(cvr_cf, conversion_proxy, reduction="none")
                - imputation_error
            )
            / propensity
        )
        debiased_cvr = 0.5 * cvr_cf + 0.5 * imputed
        predictions = torch.cat([ctr, debiased_cvr, ctcvr_direct, esmm_ctcvr], dim=-1)
        return predictions, torch.cat([ips_cvr_risk, dr_cvr_risk, propensity], dim=-1)


def build() -> nn.Module:
    """Build the compact ESCM2 model.

    Returns
    -------
    nn.Module
        Random-initialized ESCM2 module.
    """

    return ESCM2()


def example_input() -> torch.Tensor:
    """Create a small categorical recommender batch.

    Returns
    -------
    torch.Tensor
        Feature ids with shape ``(2, 6)``.
    """

    return torch.randint(0, 32, (2, 6))


MENAGERIE_ENTRIES = [
    ("ESCM2", "build", "example_input", "2022", "E5"),
]
