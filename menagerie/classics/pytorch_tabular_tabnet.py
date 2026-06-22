"""TabNet: Attentive Interpretable Tabular Learning.

Arik and Pfister, AAAI 2021.
Paper: https://ojs.aaai.org/index.php/AAAI/article/view/16826

TabNet repeatedly chooses instance-wise sparse feature masks with an attentive
transformer, applies those masks to raw tabular features, and accumulates
decision outputs across sequential decision steps.  This reconstruction keeps
the distinctive prior-update mask mechanism, GLU feature transformer, and
decision/attention split.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply sparsemax activation.

    Parameters
    ----------
    x:
        Input logits.
    dim:
        Dimension over which sparsemax is computed.

    Returns
    -------
    torch.Tensor
        Sparse probability-like tensor.
    """

    z = x - x.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    range_vals = torch.arange(1, z.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[dim] = -1
    range_vals = range_vals.view(shape)
    bound = 1 + range_vals * zs
    cumsum = torch.cumsum(zs, dim)
    is_gt = bound > cumsum
    k = is_gt.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (torch.gather(cumsum, dim, k - 1) - 1) / k.to(x.dtype)
    return torch.clamp(z - tau, min=0.0)


class GLUBlock(nn.Module):
    """Linear gated block used in TabNet feature transformers."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize a GLU projection.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        out_dim:
            Output feature dimension.
        """

        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated linear unit.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, features)``.

        Returns
        -------
        torch.Tensor
            GLU output.
        """

        h = self.bn(self.fc(x))
        a, b = h.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class TabNetStep(nn.Module):
    """One TabNet decision step with attentive masking."""

    def __init__(self, input_dim: int, decision_dim: int, attention_dim: int) -> None:
        """Initialize a decision step.

        Parameters
        ----------
        input_dim:
            Number of raw tabular features.
        decision_dim:
            Decision feature width.
        attention_dim:
            Attention feature width for the next mask.
        """

        super().__init__()
        self.transform = nn.Sequential(
            GLUBlock(input_dim, decision_dim + attention_dim),
            GLUBlock(decision_dim + attention_dim, decision_dim + attention_dim),
        )
        self.mask_proj = nn.Linear(attention_dim, input_dim)

    def forward(
        self, x: torch.Tensor, prior: torch.Tensor, attention: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one masked decision step.

        Parameters
        ----------
        x:
            Raw features with shape ``(batch, input_dim)``.
        prior:
            Feature reuse prior.
        attention:
            Attention state from the previous feature transformer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Decision output, updated attention state, and sparse mask.
        """

        mask = sparsemax(self.mask_proj(attention) * prior, dim=-1)
        masked = x * mask
        transformed = self.transform(masked)
        decision, next_attention = transformed.chunk(2, dim=-1)
        return F.relu(decision), next_attention, mask


class CompactTabNet(nn.Module):
    """Compact TabNet classifier with sequential sparse feature selection."""

    def __init__(self, input_dim: int = 10, decision_dim: int = 8, steps: int = 3) -> None:
        """Initialize TabNet steps and classifier head.

        Parameters
        ----------
        input_dim:
            Number of tabular input features.
        decision_dim:
            Decision/attention width.
        steps:
            Number of sequential decision steps.
        """

        super().__init__()
        self.initial_attention = nn.Parameter(torch.zeros(decision_dim))
        self.steps = nn.ModuleList(
            [TabNetStep(input_dim, decision_dim, decision_dim) for _ in range(steps)]
        )
        self.head = nn.Linear(decision_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a tabular batch.

        Parameters
        ----------
        x:
            Input features with shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        prior = torch.ones_like(x)
        attention = self.initial_attention.unsqueeze(0).expand(x.shape[0], -1)
        total = torch.zeros(x.shape[0], attention.shape[-1], device=x.device, dtype=x.dtype)
        for step in self.steps:
            decision, attention, mask = step(x, prior, attention)
            total = total + decision
            prior = prior * (1.5 - mask)
        return self.head(total)


def build() -> nn.Module:
    """Build a compact random-init TabNet.

    Returns
    -------
    nn.Module
        TabNet model.
    """

    return CompactTabNet()


def example_input() -> torch.Tensor:
    """Create a small tabular input batch.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(2, 10)``.
    """

    return torch.randn(2, 10)


MENAGERIE_ENTRIES = [
    ("pytorch_tabular.TabNet", "build", "example_input", "2021", "DC"),
]
