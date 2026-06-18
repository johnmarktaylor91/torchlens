"""Adaptive Logic Network (ALN), 1975, Armstrong / Godbout.

A binary tree whose leaves are linear-threshold hyperplanes and whose internal
nodes are differentiable soft-AND (smooth minimum) or soft-OR (smooth maximum)
gates.  The tree composes these into an adaptive Boolean-like decision surface
that learns via backpropagation through the smooth gate approximations.

The canonical ALN (Burrascano, 1991; Armstrong, Godbout, 1975) uses crisp
AND/OR logic; here the internal gates use the continuous relaxations:
    AND(a, b) ~ a * b                (product T-norm)
    OR(a, b)  ~ a + b - a * b        (probabilistic T-conorm)
both of which are [0,1]-valued and differentiable.

Paper: Armstrong and Godbout 1975, "The Adaptive Logic Network and Its
       Application to Pattern Recognition"; also Burrascano 1991, "Learning
       Vector Quantization for the Probabilistic Neural Network."
"""

import torch
from torch import Tensor, nn


class LinearLeaf(nn.Module):
    """Leaf node: sigmoidal linear threshold over the full input."""

    def __init__(self, in_features: int) -> None:
        """Initialize the linear threshold unit.

        Parameters
        ----------
        in_features:
            Number of input features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute sigmoid of a learned affine function.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Soft-binary activation with shape ``(batch, 1)``.
        """
        return torch.sigmoid(self.linear(x))


class ALNNode(nn.Module):
    """Internal binary tree node applying soft-AND or soft-OR to two children.

    AND gate:  y = a * b
    OR  gate:  y = a + b - a * b
    """

    def __init__(self, left: nn.Module, right: nn.Module, mode: str = "AND") -> None:
        """Initialize with two child sub-trees and a gate type.

        Parameters
        ----------
        left:
            Left child module (returns a (batch, 1) tensor in [0, 1]).
        right:
            Right child module (same shape contract).
        mode:
            Gate type: ``"AND"`` or ``"OR"``.
        """
        super().__init__()
        if mode not in ("AND", "OR"):
            raise ValueError(f"mode must be 'AND' or 'OR', got {mode!r}")
        self.left = left
        self.right = right
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate children then combine with the gate operation.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Gate output with shape ``(batch, 1)``.
        """
        a = self.left(x)
        b = self.right(x)
        if self.mode == "AND":
            return a * b
        else:
            return a + b - a * b


class AdaptiveLogicNet(nn.Module):
    """Three-level ALN: root OR -> two AND nodes -> four leaf LTUs.

    Tree structure (depth 2):
              OR
           /       \\
         AND       AND
        /   \\     /   \\
       L1   L2   L3   L4

    Four linear-threshold leaf units, two AND internal nodes, one OR root.
    This gives a compact BUT expressive Boolean decision surface over 4-dim input.
    """

    def __init__(self, in_features: int = 4) -> None:
        """Initialize a two-level ALN tree.

        Parameters
        ----------
        in_features:
            Number of input features fed to every leaf.
        """
        super().__init__()
        leaves = [LinearLeaf(in_features) for _ in range(4)]
        and1 = ALNNode(leaves[0], leaves[1], mode="AND")
        and2 = ALNNode(leaves[2], leaves[3], mode="AND")
        self.root = ALNNode(and1, and2, mode="OR")

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the ALN tree from leaves to root.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Scalar decision value in (0, 1) with shape ``(batch, 1)``.
        """
        return self.root(x)


def build() -> nn.Module:
    """Build a small two-level Adaptive Logic Network.

    Returns
    -------
    nn.Module
        Configured ``AdaptiveLogicNet`` instance.
    """
    return AdaptiveLogicNet(in_features=4)


def example_input() -> Tensor:
    """Create an example input for the ALN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [
    (
        "Adaptive Logic Network (ALN / Atree)",
        "build",
        "example_input",
        "1975",
        "RT",
    )
]
