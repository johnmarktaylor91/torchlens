"""Hierarchical Mixtures of Experts, 1994, Jordan and Jacobs, "Hierarchical Mixtures of Experts".

A tree of softmax gates routes probability mass to generalized-linear leaf experts;
the prediction is the sum over leaves weighted by root-to-leaf gate products.
"""

import torch
from torch import Tensor, nn


class HMENode(nn.Module):
    """Recursive binary HME node."""

    def __init__(self, input_size: int, out_size: int, depth: int, width: int = 2) -> None:
        """Initialize an internal gate or a leaf expert.

        Parameters
        ----------
        input_size:
            Number of input features.
        out_size:
            Number of output features.
        depth:
            Remaining tree depth; zero creates a leaf.
        width:
            Number of children for internal nodes.
        """
        super().__init__()
        self.depth = depth
        if depth == 0:
            self.expert = nn.Linear(input_size, out_size)
            self.gate = None
            self.subtrees = nn.ModuleList()
        else:
            self.expert = None
            self.gate = nn.Linear(input_size, width)
            self.subtrees = nn.ModuleList(
                [HMENode(input_size, out_size, depth - 1, width) for _ in range(width)]
            )

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate this HME subtree.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        Tensor
            Weighted subtree output.
        """
        if self.expert is not None:
            return self.expert(x)
        if self.gate is None:
            msg = "Internal HME node is missing a gate."
            raise RuntimeError(msg)
        gates = torch.softmax(self.gate(x), dim=-1)
        child_outputs = torch.stack([subtree(x) for subtree in self.subtrees], dim=1)
        return (gates.unsqueeze(-1) * child_outputs).sum(dim=1)


class HME(nn.Module):
    """Small fixed-depth hierarchical mixture of experts."""

    def __init__(self, input_size: int = 5, out_size: int = 3, depth: int = 2) -> None:
        """Initialize the HME root.

        Parameters
        ----------
        input_size:
            Number of input features.
        out_size:
            Number of output features.
        depth:
            Number of internal gating levels.
        """
        super().__init__()
        self.root = HMENode(input_size, out_size, depth)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the root HME tree.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        Tensor
            Mixture output.
        """
        return self.root(x)


def build() -> nn.Module:
    """Build a small HME soft decision tree.

    Returns
    -------
    nn.Module
        Configured ``HME`` instance.
    """
    return HME()


def example_input() -> Tensor:
    """Create a tabular float example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 5)``.
    """
    return torch.randn(2, 5)
