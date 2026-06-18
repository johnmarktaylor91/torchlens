"""OptNet, 2017, Amos and Kolter, "OptNet: Differentiable Optimization as a Layer".

Paper: Amos 2017, "OptNet: Differentiable Optimization as a Layer in Neural Networks."
This simplified QP layer solves a box-constrained positive-definite quadratic by
unrolled projected-gradient descent, preserving the differentiable optimization core.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QPLayer(nn.Module):
    """Unrolled projected-gradient quadratic-program layer."""

    def __init__(self, n_vars: int = 6, n_steps: int = 12) -> None:
        """Initialize positive-definite quadratic form.

        Parameters
        ----------
        n_vars
            Number of optimization variables.
        n_steps
            Number of unrolled solver steps.
        """
        super().__init__()
        self.n_steps = n_steps
        factor = torch.randn(n_vars, n_vars) * 0.1
        self.factor = nn.Parameter(factor)
        self.bias = nn.Parameter(torch.zeros(n_vars))

    def forward(self, q: Tensor) -> Tensor:
        """Solve a small box-constrained QP for each linear term.

        Parameters
        ----------
        q
            Linear objective vector of shape ``(batch, n_vars)``.

        Returns
        -------
        Tensor
            Approximate primal solution in ``[0, 1]``.
        """
        eye = torch.eye(self.factor.shape[0], dtype=q.dtype, device=q.device)
        qmat = self.factor.T @ self.factor + 0.2 * eye
        z = torch.full_like(q, 0.5)
        for _ in range(self.n_steps):
            grad = z @ qmat + q + self.bias
            z = torch.clamp(z - 0.15 * grad, 0.0, 1.0)
        return z


MENAGERIE_ENTRIES = [("OptNet (differentiable QP layer)", "build", "example_input", "2017", "CD")]


def build() -> nn.Module:
    """Build a simplified OptNet QP layer.

    Returns
    -------
    nn.Module
        Configured QP module.
    """
    return QPLayer()


def example_input() -> Tensor:
    """Create linear objective examples.

    Returns
    -------
    Tensor
        Example objectives with shape ``(2, 6)``.
    """
    return torch.randn(2, 6)
