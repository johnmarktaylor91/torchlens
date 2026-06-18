"""Higher-Order Input-Expansion Networks.

Two classical higher-order architectures that enrich input representations before
applying a linear readout, thereby giving the network polynomial-class decision
surfaces without additional hidden layers.

Functional Link Neural Network (FLNN), 1992:
    Pao, "Adaptive Pattern Recognition and Neural Networks."
    Functional expansion (polynomial + trig of inputs) feeds a single Linear output
    layer.  The expansion is designed/deterministic, not random (unlike RVFL).

Ridge Polynomial Neural Network (RPNN), 1995:
    Shin and Ghosh, "The pi-sigma network: An efficient higher-order neural
    network for pattern classification and function approximation."
    Cascaded Pi-Sigma units compute increasing-order ridge-polynomial terms whose
    outputs are summed to build a universal-approximation polynomial.
"""

import math

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# FLNN
# ---------------------------------------------------------------------------


class FLNN(nn.Module):
    """Functional Link Neural Network with deterministic polynomial + trig expansion.

    Expansion for input vector x of length d (``n_inputs``) with orders up to
    ``degree``:
        phi(x) = [x,  x^2, ..., x^degree,  sin(pi*x), cos(pi*x)]

    This gives ``n_inputs * (degree + 2)`` features fed into a single Linear output.
    Cross-product terms are omitted to keep the module small; the FLNN literature
    recommends either product terms or trig terms depending on the domain.
    """

    def __init__(self, n_inputs: int = 8, n_outputs: int = 1, degree: int = 2) -> None:
        """Initialize the functional-expansion linear readout.

        Parameters
        ----------
        n_inputs:
            Number of raw input features.
        n_outputs:
            Number of output targets.
        degree:
            Highest polynomial power included in the expansion (>= 1).
        """
        super().__init__()
        self.degree = degree
        # each power [1..degree] + sin + cos => (degree + 2) features per input
        expanded = n_inputs * (degree + 2)
        self.linear = nn.Linear(expanded, n_outputs)

    def _expand(self, x: Tensor) -> Tensor:
        """Apply functional expansion to input.

        Parameters
        ----------
        x:
            Raw input with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Expanded feature vector with shape ``(batch, n_inputs * (degree + 2))``.
        """
        parts = [x.pow(k) for k in range(1, self.degree + 1)]
        parts.append(torch.sin(math.pi * x))
        parts.append(torch.cos(math.pi * x))
        return torch.cat(parts, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Expand inputs then apply linear readout.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Output tensor with shape ``(batch, n_outputs)``.
        """
        return self.linear(self._expand(x))


# ---------------------------------------------------------------------------
# RPNN -- Ridge Polynomial Neural Network
# ---------------------------------------------------------------------------


class PiSigmaUnit(nn.Module):
    """Single Pi-Sigma unit computing a product of sigmoided linear projections.

    order linear projections are sigmoidally activated; their product gives the
    ridge polynomial term of the corresponding degree.
    """

    def __init__(self, in_features: int, order: int) -> None:
        """Initialize the order linear projection heads.

        Parameters
        ----------
        in_features:
            Number of input features.
        order:
            Polynomial order (number of sigma units whose product is taken).
        """
        super().__init__()
        # one Linear per sigma unit in the product
        self.projections = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(order)])

    def forward(self, x: Tensor) -> Tensor:
        """Compute the product of sigmoided projections.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Scalar output per batch element, shape ``(batch, 1)``.
        """
        # product over all sigma branches
        out = torch.sigmoid(self.projections[0](x))
        for proj in self.projections[1:]:
            out = out * torch.sigmoid(proj(x))
        return out


class RidgePolyNN(nn.Module):
    """Ridge Polynomial Neural Network: cascade of Pi-Sigma units summed.

    Unit of order k contributes x^k-class ridge-polynomial term; the cascade
    builds a polynomial of total degree ``max_order`` by additive superposition.
    A single bias is added at the output for the constant term.
    """

    def __init__(self, in_features: int = 8, max_order: int = 3) -> None:
        """Initialize the cascaded Pi-Sigma units.

        Parameters
        ----------
        in_features:
            Number of input features.
        max_order:
            Highest polynomial order; one Pi-Sigma unit per order 1..max_order.
        """
        super().__init__()
        self.units = nn.ModuleList([PiSigmaUnit(in_features, k) for k in range(1, max_order + 1)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """Sum Pi-Sigma outputs across all orders.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Scalar output per batch element, shape ``(batch, 1)``.
        """
        out = self.bias.expand(x.shape[0], 1)
        for unit in self.units:
            out = out + unit(x)
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def build_flnn() -> nn.Module:
    """Build a small FLNN with 8 inputs and degree-2 expansion.

    Returns
    -------
    nn.Module
        Configured ``FLNN`` instance.
    """
    return FLNN(n_inputs=8, n_outputs=1, degree=2)


def example_input_flnn() -> Tensor:
    """Create an example input for the FLNN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


def build_rpnn() -> nn.Module:
    """Build a small Ridge Polynomial Neural Network with max order 3.

    Returns
    -------
    nn.Module
        Configured ``RidgePolyNN`` instance.
    """
    return RidgePolyNN(in_features=8, max_order=3)


def example_input_rpnn() -> Tensor:
    """Create an example input for the RPNN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    (
        "Functional Link Neural Network (FLNN, Pao)",
        "build_flnn",
        "example_input_flnn",
        "1992",
        "RT",
    ),
    (
        "Ridge Polynomial Neural Network (RPNN)",
        "build_rpnn",
        "example_input_rpnn",
        "1995",
        "RT",
    ),
]
