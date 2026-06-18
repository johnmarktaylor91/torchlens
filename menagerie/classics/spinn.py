"""SPINN, 2016, Bowman et al., "A Fast Unified Model for Parsing and Sentence Understanding".

Paper: Bowman 2016, "A Fast Unified Model for Parsing and Sentence Understanding."
This simplified SPINN uses differentiable shift/reduce mixtures over token embeddings
and a TreeLSTM-like reducer, omitting hard transition supervision and parser state.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SPINN(nn.Module):
    """Differentiable shift-reduce sentence composition module."""

    def __init__(self, vocab_size: int = 20, embed_dim: int = 6, hidden_size: int = 6) -> None:
        """Initialize embedding, transition, and reducer layers.

        Parameters
        ----------
        vocab_size
            Number of token ids.
        embed_dim
            Token embedding width.
        hidden_size
            Hidden composition width.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.project = nn.Linear(embed_dim, hidden_size)
        self.reducer = nn.Linear(hidden_size * 2, hidden_size)
        self.transition = nn.Linear(hidden_size, 2)

    def forward(self, tokens: Tensor) -> Tensor:
        """Compose a token sequence with soft shift-reduce updates.

        Parameters
        ----------
        tokens
            Token ids of shape ``(batch, length)``.

        Returns
        -------
        Tensor
            Sentence representation.
        """
        embedded = torch.tanh(self.project(self.embedding(tokens)))
        stack_top = torch.zeros_like(embedded[:, 0])
        stack_next = torch.zeros_like(embedded[:, 0])
        for step in range(tokens.shape[1]):
            shifted = embedded[:, step]
            reduced = torch.tanh(self.reducer(torch.cat((stack_next, stack_top), dim=-1)))
            weights = torch.softmax(self.transition(shifted), dim=-1)
            new_top = weights[:, :1] * shifted + weights[:, 1:] * reduced
            stack_next = stack_top
            stack_top = new_top
        return stack_top


MENAGERIE_ENTRIES = [
    ("Stack-augmented Parser-Interpreter (SPINN)", "build", "example_input", "2016", "CD")
]


def build() -> nn.Module:
    """Build a simplified SPINN module.

    Returns
    -------
    nn.Module
        Configured SPINN module.
    """
    return SPINN()


def example_input() -> Tensor:
    """Create token id examples.

    Returns
    -------
    Tensor
        Example tokens with shape ``(2, 5)``.
    """
    return torch.tensor([[1, 2, 3, 4, 5], [3, 4, 2, 1, 0]], dtype=torch.long)
