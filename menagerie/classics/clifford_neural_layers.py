"""Compact Clifford neural layer with geometric-product channel mixing.

Clifford neural layers generalize real/complex/quaternion convolutions by storing
features over geometric-algebra blades and mixing them with a metric-dependent
Clifford/geometric product.  The distinctive primitive is blade-wise convolution
followed by the signed multiplication table induced by the algebra signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mul_blades(a: int, b: int, metric: tuple[int, int]) -> tuple[int, float]:
    """Multiply two basis blades in a diagonal Clifford algebra.

    Parameters
    ----------
    a:
        First blade bit mask.
    b:
        Second blade bit mask.
    metric:
        Diagonal metric signs.

    Returns
    -------
    tuple[int, float]
        Output blade mask and sign.
    """

    sign = 1.0
    for i in range(len(metric)):
        if (a >> i) & 1:
            swaps = (b & ((1 << i) - 1)).bit_count()
            sign *= -1.0 if swaps % 2 else 1.0
    common = a & b
    for i, m in enumerate(metric):
        if (common >> i) & 1:
            sign *= float(m)
    return a ^ b, sign


class CliffordLinear(nn.Module):
    """Linear layer whose channels are multivectors mixed by geometric product."""

    def __init__(
        self, in_channels: int, out_channels: int, metric: tuple[int, int] = (1, 1)
    ) -> None:
        """Initialize blade kernels and multiplication table.

        Parameters
        ----------
        in_channels:
            Scalar channel count per blade.
        out_channels:
            Output scalar channel count per blade.
        metric:
            Clifford algebra signature.
        """

        super().__init__()
        self.blades = 2 ** len(metric)
        self.weight = nn.Parameter(torch.randn(self.blades, out_channels, in_channels) * 0.05)
        table = torch.zeros(self.blades, self.blades, self.blades)
        for left in range(self.blades):
            for right in range(self.blades):
                out, sign = _mul_blades(left, right, metric)
                table[left, right, out] = sign
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Clifford geometric-product channel mixing.

        Parameters
        ----------
        x:
            Multivector input ``(B, blades, Cin)``.

        Returns
        -------
        torch.Tensor
            Multivector output ``(B, blades, Cout)``.
        """

        projected = torch.einsum("bli,roi->blro", x, self.weight)
        return torch.einsum("blro,lrs->bso", projected, self.table)


class CliffordCompact(nn.Module):
    """Small Clifford MLP over two-dimensional Euclidean algebra blades."""

    def __init__(self) -> None:
        """Initialize two geometric-product layers."""

        super().__init__()
        self.layer1 = CliffordLinear(6, 8)
        self.layer2 = CliffordLinear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the compact Clifford network.

        Parameters
        ----------
        x:
            Multivector input.

        Returns
        -------
        torch.Tensor
            Multivector output.
        """

        return self.layer2(torch.tanh(self.layer1(x)))


def build() -> nn.Module:
    """Build a compact Clifford neural network.

    Returns
    -------
    nn.Module
        Clifford multivector network.
    """

    return CliffordCompact()


def example_input() -> torch.Tensor:
    """Create multivector features with four blades.

    Returns
    -------
    torch.Tensor
        Example tensor ``(2, 4, 6)``.
    """

    return torch.randn(2, 4, 6)


MENAGERIE_ENTRIES = [
    (
        "Clifford neural layers (geometric-product multivector mixing)",
        "build",
        "example_input",
        "2022",
        "DC",
    ),
]
