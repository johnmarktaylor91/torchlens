"""Neural Logic Machine, 2019, Dong et al., "Neural Logic Machines".

Paper: Dong 2019, "Neural Logic Machines."
Binary predicate tensors are transformed by existential and universal reductions,
pairwise predicate composition, and an MLP-style logic layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuralLogicMachine(nn.Module):
    """One-layer binary-predicate Neural Logic Machine."""

    def __init__(self, n_predicates: int = 4, hidden: int = 6) -> None:
        """Initialize binary predicate transformation.

        Parameters
        ----------
        n_predicates
            Number of input binary predicates.
        hidden
            Hidden output predicate count.
        """
        super().__init__()
        self.logic = nn.Sequential(
            nn.Linear(n_predicates * 4, hidden), nn.ReLU(), nn.Linear(hidden, n_predicates)
        )

    def forward(self, binary: Tensor) -> Tensor:
        """Apply quantifier-style reductions and binary predicate composition.

        Parameters
        ----------
        binary
            Binary predicate probabilities of shape ``(batch, objects, objects, predicates)``.

        Returns
        -------
        Tensor
            Updated binary predicate probabilities.
        """
        exists_row = binary.amax(dim=2, keepdim=True).expand_as(binary)
        exists_col = binary.amax(dim=1, keepdim=True).expand_as(binary)
        forall_row = binary.amin(dim=2, keepdim=True).expand_as(binary)
        features = torch.cat((binary, exists_row, exists_col, forall_row), dim=-1)
        return torch.sigmoid(self.logic(features))


MENAGERIE_ENTRIES = [("Neural Logic Machine (NLM)", "build", "example_input", "2019", "CD")]


def build() -> nn.Module:
    """Build a compact Neural Logic Machine layer.

    Returns
    -------
    nn.Module
        Configured NLM module.
    """
    return NeuralLogicMachine()


def example_input() -> Tensor:
    """Create binary predicate tensor examples.

    Returns
    -------
    Tensor
        Example predicate tensor with shape ``(2, 4, 4, 4)``.
    """
    return torch.rand(2, 4, 4, 4)
