"""Neuro-Fuzzy Inference Modules.

ANFIS, 1993, Jang, "ANFIS: Adaptive-Network-Based Fuzzy Inference System."
    Gaussian membership functions feed product rule firing strengths, normalized
    rule weights, first-order Takagi-Sugeno consequents, and a weighted output sum.
    Only the forward inference graph is included; hybrid least-squares training is omitted.

Neo-Fuzzy Neuron, 1992, Yamakawa et al., "A neo fuzzy neuron and its
    applications to system identification and prediction of the system behaviour."
    Per-input piecewise-linear (triangular) fuzzy membership banks multiplied by
    tunable consequent weights, summed across inputs and membership functions.
    This implements the zero-order Takagi-Sugeno inference of the NFN in a single
    differentiable layer that is linear in its parameters.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn


class ANFIS(nn.Module):
    """Small differentiable Takagi-Sugeno neuro-fuzzy inference system."""

    def __init__(self, n_inputs: int = 2, n_mfs: int = 3) -> None:
        """Initialize Gaussian membership functions and consequents.

        Parameters
        ----------
        n_inputs:
            Number of input variables.
        n_mfs:
            Number of membership functions per input.
        """
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, n_mfs).repeat(n_inputs, 1)
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        rules = torch.tensor(
            list(itertools.product(range(n_mfs), repeat=n_inputs)), dtype=torch.long
        )
        self.register_buffer("rule_indices", rules)
        self.consequents = nn.Parameter(torch.randn(rules.shape[0], n_inputs + 1) * 0.1)

    def _memberships(self, x: Tensor) -> Tensor:
        """Compute Gaussian membership degrees.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Membership tensor with shape ``(batch, n_inputs, n_mfs)``.
        """
        widths = torch.nn.functional.softplus(self.log_widths) + 0.15
        diff = x[:, :, None] - self.centers[None, :, :]
        return torch.exp(-0.5 * (diff / widths[None, :, :]).pow(2))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the five ANFIS inference layers.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, 2)``.

        Returns
        -------
        Tensor
            Crisp Takagi-Sugeno output with shape ``(batch, 1)``.
        """
        memberships = self._memberships(x)
        firing = x.new_ones(x.shape[0], self.rule_indices.shape[0])
        for dim in range(x.shape[1]):
            firing = firing * memberships[:, dim, self.rule_indices[:, dim]]
        normalized = firing / firing.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        augmented = torch.cat((x, torch.ones_like(x[:, :1])), dim=-1)
        rule_outputs = augmented @ self.consequents.T
        return (normalized * rule_outputs).sum(dim=-1, keepdim=True)


def build() -> nn.Module:
    """Build a small ANFIS module.

    Returns
    -------
    nn.Module
        Configured ``ANFIS`` instance.
    """
    return ANFIS()


def example_input() -> Tensor:
    """Create an ANFIS input example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 2)``.
    """
    return torch.tensor([[0.25, -0.5]], dtype=torch.float32)


MENAGERIE_ENTRIES = [
    ("ANFIS (Adaptive Neuro-Fuzzy Inference System)", "build", "example_input", "1993", "MB1"),
    ("Neo-Fuzzy Neuron (Yamakawa)", "build_nfn", "example_input_nfn", "1992", "RT"),
]


# ---------------------------------------------------------------------------
# Neo-Fuzzy Neuron (Yamakawa 1992)
# ---------------------------------------------------------------------------


class NeoFuzzyNeuron(nn.Module):
    """Per-input triangular-membership fuzzy nonlinear synapse, summed to output.

    For each input x_i the module computes n_mfs overlapping triangular membership
    functions mu_ij(x_i), multiplies each by a learnable consequent weight w_ij,
    and sums over all inputs and membership functions:

        y = sum_i sum_j w_ij * mu_ij(x_i)

    Because the triangular MFs form a partition of unity (sum to 1 for any x_i in
    the covered range), the output is a linear combination of the weights: the
    network is linear in its parameters, making it very easy to train.

    Triangular MFs are implemented via ReLU differences (traceable, no branching):
        mu(x, c_l, c, c_r) = relu(1 - |x - c| / half_width)  clipped to [0, 1]
    """

    def __init__(
        self, n_inputs: int = 4, n_mfs: int = 5, x_min: float = -1.0, x_max: float = 1.0
    ) -> None:
        """Initialize uniformly-spaced triangular MF centres and consequent weights.

        Parameters
        ----------
        n_inputs:
            Number of input variables.
        n_mfs:
            Number of triangular membership functions per input (>= 2).
        x_min:
            Lower bound of the input range.
        x_max:
            Upper bound of the input range.
        """
        super().__init__()
        # Uniformly-spaced centres covering [x_min, x_max]
        centres = torch.linspace(x_min, x_max, n_mfs)  # (n_mfs,)
        self.register_buffer("centres", centres)
        # Half-width = spacing between adjacent centres
        half_width = (x_max - x_min) / (n_mfs - 1) if n_mfs > 1 else 1.0
        self.half_width = half_width
        # One learnable consequent weight per (input, MF) pair
        self.weights = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    def forward(self, x: Tensor) -> Tensor:
        """Compute neuro-fuzzy neuron output.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Scalar output with shape ``(batch, 1)``.
        """
        # x: (B, n_inputs) -> (B, n_inputs, 1) vs centres (n_mfs,) -> (1, 1, n_mfs)
        diff = x[:, :, None] - self.centres[None, None, :]  # (B, n_inputs, n_mfs)
        # Triangular MF: relu(1 - |diff| / half_width), clamped to [0, 1]
        mu = torch.relu(1.0 - diff.abs() / self.half_width)  # (B, n_inputs, n_mfs)
        # Weighted sum: mu * w, sum over inputs and MFs -> scalar
        out = (mu * self.weights[None, :, :]).sum(dim=(-1, -2), keepdim=False)
        return out.unsqueeze(-1)  # (B, 1)


def build_nfn() -> nn.Module:
    """Build a small Neo-Fuzzy Neuron.

    Returns
    -------
    nn.Module
        Configured ``NeoFuzzyNeuron`` instance.
    """
    return NeoFuzzyNeuron(n_inputs=4, n_mfs=5)


def example_input_nfn() -> Tensor:
    """Create an input example for the Neo-Fuzzy Neuron.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.rand(1, 4) * 2.0 - 1.0
