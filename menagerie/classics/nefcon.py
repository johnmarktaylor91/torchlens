"""NEFCON / NEFCLASS neuro-fuzzy model, 1994/1999, Nauck and Kruse.

Paper: Nauck and Kruse 1994, "NEFCON-I: An XCON-like neuro-fuzzy controller."
Gaussian input memberships fire differentiable fuzzy rules whose normalized
strengths aggregate trainable consequents.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    (
        "NEFCON / NEFCLASS (neuro-fuzzy controller/classifier)",
        "build",
        "example_input",
        "1994/1999",
        "CF",
    )
]


class NEFCON(nn.Module):
    """Gaussian-membership neuro-fuzzy rule aggregator."""

    def __init__(
        self,
        n_inputs: int = 3,
        terms_per_input: int = 3,
        n_rules: int = 6,
        n_outputs: int = 2,
    ) -> None:
        """Initialize memberships, rule antecedents, and consequents.

        Parameters
        ----------
        n_inputs
            Number of input variables.
        terms_per_input
            Number of Gaussian terms per variable.
        n_rules
            Number of fuzzy rules.
        n_outputs
            Number of output dimensions/classes.
        """
        super().__init__()
        centers = torch.linspace(0.0, 1.0, terms_per_input).repeat(n_inputs, 1)
        self.centers = nn.Parameter(centers)
        self.log_scales = nn.Parameter(torch.full((n_inputs, terms_per_input), -1.2))
        self.register_buffer(
            "antecedents",
            torch.stack(
                [torch.arange(n_inputs) * 0 + (rule % terms_per_input) for rule in range(n_rules)]
            ),
        )
        self.consequents = nn.Parameter(torch.randn(n_rules, n_outputs) * 0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate normalized fuzzy rules.

        Parameters
        ----------
        x
            Input variables of shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Aggregated consequent output.
        """
        scales = self.log_scales.exp().clamp_min(1.0e-3)
        memberships = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / scales) ** 2)
        rule_terms = []
        for rule in range(self.antecedents.shape[0]):
            selected = memberships.gather(
                2,
                self.antecedents[rule].view(1, -1, 1).expand(x.shape[0], -1, 1),
            ).squeeze(-1)
            rule_terms.append(selected.prod(dim=-1))
        firing = torch.stack(rule_terms, dim=-1)
        norm = firing / firing.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return norm @ self.consequents


def build() -> nn.Module:
    """Build a compact NEFCON/NEFCLASS module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return NEFCON()


def example_input() -> Tensor:
    """Return bounded neuro-fuzzy inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 3)``.
    """
    return torch.rand(2, 3)
