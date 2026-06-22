"""ChebyKAN compact Chebyshev-polynomial Kolmogorov-Arnold Network.

ChebyKAN replaces spline edge functions in KANs with learnable Chebyshev
polynomial expansions on each input-output edge.  The load-bearing primitive is
the per-edge univariate function ``sum_k c[o, i, k] T_k(tanh(x_i))`` followed by
summation over input dimensions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    """KAN layer using Chebyshev basis functions on every edge."""

    def __init__(self, in_features: int, out_features: int, degree: int = 5) -> None:
        """Initialize Chebyshev coefficients.

        Parameters
        ----------
        in_features:
            Number of input coordinates.
        out_features:
            Number of output coordinates.
        degree:
            Highest Chebyshev degree.
        """

        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(out_features, in_features, degree + 1) * 0.05)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Chebyshev basis ``T_0..T_degree``.

        Parameters
        ----------
        x:
            Input tensor ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Basis tensor ``(..., in_features, degree + 1)``.
        """

        z = torch.tanh(x)
        terms = [torch.ones_like(z), z]
        for _ in range(2, self.degree + 1):
            terms.append(2.0 * z * terms[-1] - terms[-2])
        return torch.stack(terms[: self.degree + 1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply edge-wise Chebyshev expansions and sum incoming edges.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Output features.
        """

        return torch.einsum("...ik,oik->...o", self._basis(x), self.coeff)


class ChebyKANCompact(nn.Module):
    """Two-layer ChebyKAN classifier/regressor."""

    def __init__(self) -> None:
        """Initialize a compact ChebyKAN network."""

        super().__init__()
        self.layer1 = ChebyKANLayer(8, 16, degree=5)
        self.layer2 = ChebyKANLayer(16, 4, degree=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the ChebyKAN.

        Parameters
        ----------
        x:
            Input tensor ``(B, 8)``.

        Returns
        -------
        torch.Tensor
            Output tensor ``(B, 4)``.
        """

        return self.layer2(torch.tanh(self.layer1(x)))


def build() -> nn.Module:
    """Build a compact ChebyKAN.

    Returns
    -------
    nn.Module
        ChebyKAN model.
    """

    return ChebyKANCompact()


def example_input() -> torch.Tensor:
    """Create tabular input for ChebyKAN.

    Returns
    -------
    torch.Tensor
        Example tensor ``(2, 8)``.
    """

    return torch.randn(2, 8)


MENAGERIE_ENTRIES = [
    ("ChebyKAN (Chebyshev polynomial edge-function KAN)", "build", "example_input", "2024", "DC"),
]
