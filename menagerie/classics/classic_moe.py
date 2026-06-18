"""Classic Adaptive Mixture of Experts, 1991, Jacobs, Jordan, Nowlan, and Hinton.

Several dense experts are combined by a softmax gating network; every expert is
evaluated and the output is the gate-probability-weighted sum.
"""

import torch
from torch import Tensor, nn


class ClassicMoE(nn.Module):
    """Dense soft-gated mixture of small MLP experts."""

    def __init__(
        self, input_size: int = 5, hidden_size: int = 6, out_size: int = 3, n_experts: int = 4
    ) -> None:
        """Initialize experts and the gating network.

        Parameters
        ----------
        input_size:
            Number of input features.
        hidden_size:
            Hidden width inside each expert.
        out_size:
            Number of output features.
        n_experts:
            Number of dense experts.
        """
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, out_size)
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Linear(input_size, n_experts)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate all experts and combine them with soft gate probabilities.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        Tensor
            Mixture output of shape ``(batch, out_size)``.
        """
        gates = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (gates.unsqueeze(-1) * expert_outputs).sum(dim=1)


def build() -> nn.Module:
    """Build a small dense mixture-of-experts module.

    Returns
    -------
    nn.Module
        Configured ``ClassicMoE`` instance.
    """
    return ClassicMoE()


def example_input() -> Tensor:
    """Create a tabular float example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 5)``.
    """
    return torch.randn(2, 5)
