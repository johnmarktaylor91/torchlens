"""Hyperbolic neural networks on the Poincare ball, 2018-2019.

Paper: Ganea, Becigneul, and Hofmann 2018, "Hyperbolic Neural Networks";
Chami et al. 2019, "Hyperbolic Graph Convolutional Neural Networks".
The modules implement Poincare-ball exp/log maps, Mobius addition, and small
hyperbolic linear/aggregation layers with clamped numerics for tracing.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _lambda_x(x: Tensor, c: float, eps: float = 1e-5) -> Tensor:
    """Compute the conformal factor on the Poincare ball.

    Parameters
    ----------
    x
        Ball point tensor.
    c
        Positive curvature magnitude.
    eps
        Numerical clamp.

    Returns
    -------
    Tensor
        Conformal factor.
    """
    norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp_max((1.0 - eps) / c)
    return 2.0 / (1.0 - c * norm_sq).clamp_min(eps)


def _project(x: Tensor, c: float, eps: float = 1e-5) -> Tensor:
    """Project points inside the Poincare ball.

    Parameters
    ----------
    x
        Candidate point tensor.
    c
        Positive curvature magnitude.
    eps
        Numerical margin from the boundary.

    Returns
    -------
    Tensor
        Projected point tensor.
    """
    max_norm = (1.0 - eps) / math.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def expmap0(u: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Map tangent vectors at the origin to the Poincare ball.

    Parameters
    ----------
    u
        Tangent vector tensor.
    c
        Positive curvature magnitude.
    eps
        Numerical clamp.

    Returns
    -------
    Tensor
        Ball point tensor.
    """
    sqrt_c = math.sqrt(c)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    mapped = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    return _project(mapped, c, eps)


def logmap0(x: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Map Poincare-ball points to the tangent space at the origin.

    Parameters
    ----------
    x
        Ball point tensor.
    c
        Positive curvature magnitude.
    eps
        Numerical clamp.

    Returns
    -------
    Tensor
        Tangent vector tensor.
    """
    sqrt_c = math.sqrt(c)
    projected = _project(x, c, eps)
    norm = projected.norm(dim=-1, keepdim=True).clamp_min(eps)
    arg = (sqrt_c * norm).clamp(max=1.0 - eps)
    return torch.atanh(arg) * projected / (sqrt_c * norm)


def mobius_add(x: Tensor, y: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Add two Poincare-ball points with Mobius addition.

    Parameters
    ----------
    x
        First point tensor.
    y
        Second point tensor.
    c
        Positive curvature magnitude.
    eps
        Numerical clamp.

    Returns
    -------
    Tensor
        Mobius sum.
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denominator = (1.0 + 2.0 * c * xy + c * c * x2 * y2).clamp_min(eps)
    return _project(numerator / denominator, c, eps)


def mobius_matvec(weight: Tensor, x: Tensor, c: float = 1.0) -> Tensor:
    """Apply a linear map through origin log/exp maps.

    Parameters
    ----------
    weight
        Euclidean matrix with shape ``(out, in)``.
    x
        Ball point tensor with shape ``(..., in)``.
    c
        Positive curvature magnitude.

    Returns
    -------
    Tensor
        Ball point tensor with output dimensionality.
    """
    tangent = logmap0(x, c)
    return expmap0(tangent @ weight.T, c)


class HyperbolicLinear(nn.Module):
    """Poincare-ball Mobius linear layer with hyperbolic bias."""

    def __init__(self, d_in: int, d_out: int, c: float = 1.0) -> None:
        """Initialize hyperbolic linear parameters.

        Parameters
        ----------
        d_in
            Input dimensionality.
        d_out
            Output dimensionality.
        c
            Positive curvature magnitude.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        """Apply Mobius matrix-vector multiplication and bias addition.

        Parameters
        ----------
        x
            Ball point tensor.

        Returns
        -------
        Tensor
            Ball point tensor.
        """
        out = mobius_matvec(self.weight, x, self.c)
        return mobius_add(out, expmap0(self.bias.expand_as(out), self.c), self.c)


class HyperbolicMLP(nn.Module):
    """Small hyperbolic MLP with tangent-space nonlinearities."""

    def __init__(self, d_in: int = 4, d_hidden: int = 6, d_out: int = 2, c: float = 1.0) -> None:
        """Initialize hyperbolic MLP layers.

        Parameters
        ----------
        d_in
            Input tangent dimensionality.
        d_hidden
            Hidden dimensionality.
        d_out
            Output tangent dimensionality.
        c
            Positive curvature magnitude.
        """
        super().__init__()
        self.first = HyperbolicLinear(d_in, d_hidden, c)
        self.second = HyperbolicLinear(d_hidden, d_out, c)
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        """Map tangent inputs to the ball and apply hyperbolic layers.

        Parameters
        ----------
        x
            Tangent input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Tangent-space output logits.
        """
        h = self.first(expmap0(x, self.c))
        h = expmap0(torch.relu(logmap0(h, self.c)), self.c)
        return logmap0(self.second(h), self.c)


class HyperbolicGCN(nn.Module):
    """Two-layer Poincare-ball graph convolution sketch."""

    def __init__(self, d_in: int = 4, d_hidden: int = 5, d_out: int = 2, c: float = 1.0) -> None:
        """Initialize hyperbolic graph layers.

        Parameters
        ----------
        d_in
            Input tangent dimensionality.
        d_hidden
            Hidden dimensionality.
        d_out
            Output tangent dimensionality.
        c
            Positive curvature magnitude.
        """
        super().__init__()
        self.first = HyperbolicLinear(d_in, d_hidden, c)
        self.second = HyperbolicLinear(d_hidden, d_out, c)
        self.c = c

    def _aggregate(self, x: Tensor, adj: Tensor) -> Tensor:
        """Aggregate neighbors in tangent space and return to the ball.

        Parameters
        ----------
        x
            Ball point tensor with shape ``(nodes, features)``.
        adj
            Adjacency matrix with shape ``(nodes, nodes)``.

        Returns
        -------
        Tensor
            Aggregated ball point tensor.
        """
        weights = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        tangent = weights @ logmap0(x, self.c)
        return expmap0(tangent, self.c)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Apply hyperbolic graph convolution to node features.

        Parameters
        ----------
        inputs
            Tuple of tangent node features and adjacency matrix.

        Returns
        -------
        Tensor
            Tangent-space node outputs.
        """
        features, adj = inputs
        h = self.first(expmap0(features, self.c))
        h = self._aggregate(h, adj)
        h = expmap0(torch.relu(logmap0(h, self.c)), self.c)
        h = self.second(self._aggregate(h, adj))
        return logmap0(h, self.c)


def build_hyperbolic_mlp() -> nn.Module:
    """Build a hyperbolic MLP.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return HyperbolicMLP()


def example_input_hyperbolic_mlp() -> Tensor:
    """Return an example tangent input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 4) * 0.1


def build_hyperbolic_gcn() -> nn.Module:
    """Build a hyperbolic GCN.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return HyperbolicGCN()


def example_input_hyperbolic_gcn() -> tuple[Tensor, Tensor]:
    """Return example node features and adjacency.

    Returns
    -------
    tuple[Tensor, Tensor]
        Feature and adjacency tensors.
    """
    features = torch.randn(5, 4) * 0.1
    adj = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    return features, adj


MENAGERIE_ENTRIES = [
    (
        "Hyperbolic MLP (Poincare-ball Mobius layers)",
        "build_hyperbolic_mlp",
        "example_input_hyperbolic_mlp",
        "2018",
        "CH-C",
    ),
    (
        "Hyperbolic GCN (HGCN)",
        "build_hyperbolic_gcn",
        "example_input_hyperbolic_gcn",
        "2019",
        "CH-C",
    ),
]
