"""Arrow Language Model, 2026, Paul Tarau.

Paper: Tarau 2026, "Arrow Language Models."
Token prefixes are folded through a learned implication-like bilinear operator and read out
as next-token logits; proof search and formal type checking are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ArrowLanguageModel(nn.Module):
    """Left-nested learned implication-chain language model."""

    def __init__(self, vocab_size: int = 128, dim: int = 32) -> None:
        """Initialize token embeddings, arrow operator, and readout.

        Parameters
        ----------
        vocab_size
            Number of token ids.
        dim
            Embedding width.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.readout = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Fold tokens into an implication context and predict next-token logits.

        Parameters
        ----------
        tokens
            Token ids with shape ``(batch, 64)``.

        Returns
        -------
        Tensor
            Next-token logits.
        """
        emb = self.embedding(tokens)
        context = emb[:, 0]
        states: list[Tensor] = []
        for step in range(1, emb.shape[1]):
            token = emb[:, step]
            arrow = torch.tanh(self.left(context) * self.right(token))
            gate = torch.sigmoid(self.gate(torch.cat((context, token), dim=-1)))
            context = gate * arrow + (1.0 - gate) * context
            states.append(context)
        return self.readout(torch.stack(states, dim=1))


MENAGERIE_ENTRIES = [("Arrow Language Model", "build", "example_input", "2026", "DA")]


def build() -> nn.Module:
    """Build an Arrow Language Model.

    Returns
    -------
    nn.Module
        Configured Arrow LM.
    """
    return ArrowLanguageModel()


def example_input() -> Tensor:
    """Create token ids.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.randint(0, 128, (1, 64), dtype=torch.long)
