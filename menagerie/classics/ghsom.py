"""Growing Hierarchical SOM, 2000, Dittenbach, Merkl, and Rauber.

Paper: Dittenbach et al. 2000, "The growing hierarchical self-organizing map."
GHSOM grows maps and child maps by quantization-error thresholds. This
trace-clean simplified module keeps one parent map and one child map, exposing
the topology-preserving soft BMU responses rather than mutating hierarchy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("GHSOM (Growing Hierarchical SOM)", "build", "example_input", "2000", "CF")]


class GHSOM(nn.Module):
    """Simplified two-level hierarchical SOM response module."""

    def __init__(self, dim: int = 4, parent_side: int = 3, child_side: int = 2) -> None:
        """Initialize parent and child map prototypes.

        Parameters
        ----------
        dim
            Input dimensionality.
        parent_side
            Width and height of the parent map grid.
        child_side
            Width and height of the child map grid.
        """
        super().__init__()
        self.parent = nn.Parameter(torch.randn(parent_side * parent_side, dim))
        self.child = nn.Parameter(torch.randn(child_side * child_side, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Compute parent and child soft BMU responses.

        Parameters
        ----------
        x
            Input samples of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Concatenated parent response, child response, and reconstruction error.
        """
        parent_dist = torch.cdist(x, self.parent)
        parent_resp = torch.softmax(-parent_dist * 4.0, dim=-1)
        parent_recon = parent_resp @ self.parent
        residual = x - parent_recon
        child_dist = torch.cdist(residual, self.child)
        child_resp = torch.softmax(-child_dist * 4.0, dim=-1)
        child_recon = child_resp @ self.child
        error = ((residual - child_recon) ** 2).mean(dim=-1, keepdim=True)
        return torch.cat((parent_resp, child_resp, error), dim=-1)


def build() -> nn.Module:
    """Build a simplified GHSOM response module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return GHSOM()


def example_input() -> Tensor:
    """Return vector samples.

    Returns
    -------
    Tensor
        Example tensor of shape ``(4, 4)``.
    """
    return torch.randn(4, 4)
