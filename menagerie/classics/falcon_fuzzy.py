"""FALCON fuzzy adaptive learning control network, 1991, Lin and Lee.

Paper: Lin and Lee 1991, "Neural-network-based fuzzy logic control and decision
system." FALCON implements a five-layer Mamdani fuzzy controller with input
memberships, rule firing, output memberships, and center-of-gravity
defuzzification.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("FALCON (Fuzzy Adaptive Learning Control Network)", "build", "example_input", "1991", "CF")
]


class FALCONFuzzy(nn.Module):
    """Mamdani-style neuro-fuzzy controller with Gaussian terms."""

    def __init__(self, n_inputs: int = 3, n_rules: int = 5, n_outputs: int = 2) -> None:
        """Initialize membership and consequent centers.

        Parameters
        ----------
        n_inputs
            Number of controller inputs.
        n_rules
            Number of fuzzy rules.
        n_outputs
            Number of defuzzified outputs.
        """
        super().__init__()
        self.input_centers = nn.Parameter(torch.rand(n_rules, n_inputs))
        self.log_widths = nn.Parameter(torch.full((n_rules, n_inputs), -1.0))
        self.output_centers = nn.Parameter(
            torch.linspace(-1.0, 1.0, n_rules).view(n_rules, 1).repeat(1, n_outputs)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Defuzzify Mamdani rule consequents.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Defuzzified controller output.
        """
        width = self.log_widths.exp().clamp_min(1.0e-3)
        membership = torch.exp(-0.5 * ((x.unsqueeze(1) - self.input_centers) / width) ** 2)
        firing = membership.prod(dim=-1)
        numerator = firing @ self.output_centers
        denominator = firing.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return numerator / denominator


def build() -> nn.Module:
    """Build a compact FALCON controller.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return FALCONFuzzy()


def example_input() -> Tensor:
    """Return bounded controller inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 3)``.
    """
    return torch.rand(2, 3)
