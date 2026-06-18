"""Marr-Albus-Ito cerebellar perceptron, 1969, Marr, Albus, and Ito.

Paper: "A theory of cerebellar cortex." Mossy-fiber inputs expand through sparse
granule-cell activations and feed a Purkinje perceptron readout; climbing-fiber LTD
training is omitted from this forward-only substrate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Marr-Albus-Ito Cerebellar Perceptron", "build", "example_input", "1969", "DB")
]


class CerebellarPerceptron(nn.Module):
    """Sparse granule expansion feeding a Purkinje readout."""

    def __init__(self, mf_dim: int = 64, gc_dim: int = 256, n_out: int = 8, k: int = 16) -> None:
        """Initialize granule expansion and Purkinje readout.

        Parameters
        ----------
        mf_dim
            Mossy-fiber input dimensionality.
        gc_dim
            Granule-cell expansion dimensionality.
        n_out
            Purkinje output dimensionality.
        k
            Number of winning granule cells.
        """
        super().__init__()
        self.granule = nn.Linear(mf_dim, gc_dim)
        self.purkinje = nn.Linear(gc_dim, n_out)
        self.k = k

    def forward(self, mf: Tensor) -> Tensor:
        """Compute sparse granule code and Purkinje output.

        Parameters
        ----------
        mf
            Mossy-fiber input of shape ``(batch, mf_dim)``.

        Returns
        -------
        Tensor
            Purkinje readout activity.
        """
        granule = torch.relu(self.granule(mf))
        values, _ = torch.topk(granule, self.k, dim=-1)
        threshold = values[..., -1:].expand_as(granule)
        sparse = granule * (granule >= threshold).to(granule.dtype)
        return self.purkinje(sparse)


def build() -> nn.Module:
    """Build a small cerebellar perceptron module.

    Returns
    -------
    nn.Module
        Configured ``CerebellarPerceptron`` instance.
    """
    return CerebellarPerceptron()


def example_input() -> Tensor:
    """Return a mossy-fiber vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)
