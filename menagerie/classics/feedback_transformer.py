"""Feedback Transformer with higher-level recurrent memory.

Paper: "Addressing Some Limitations of Transformers with Feedback Memory",
Fan et al., TACL 2021.

The Feedback Transformer processes tokens sequentially. Instead of each layer
attending to same-level past states, the next token's low-level representation
attends to the highest-level representations from earlier timesteps.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FeedbackBlock(nn.Module):
    """Single feedback-attention block."""

    def __init__(self, d_model: int) -> None:
        """Initialize projections, normalization, and feed-forward layers.

        Parameters
        ----------
        d_model:
            Hidden width.
        """
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """Attend from current state to feedback memory.

        Parameters
        ----------
        x:
            Current token state with shape ``(batch, d_model)``.
        memory:
            Previous top-layer states with shape ``(batch, time, d_model)``.

        Returns
        -------
        Tensor
            Updated current state with shape ``(batch, d_model)``.
        """
        q = self.q(x).unsqueeze(1)
        scores = torch.matmul(q, self.k(memory).transpose(-1, -2)) / math.sqrt(x.shape[-1])
        context = torch.matmul(torch.softmax(scores, dim=-1), self.v(memory)).squeeze(1)
        x = self.norm_attn(x + self.out(context))
        return self.norm_ff(x + self.ff(x))


class FeedbackTransformerLM(nn.Module):
    """Compact sequential Feedback Transformer language model."""

    def __init__(self, vocab_size: int = 96, d_model: int = 32, n_layers: int = 2) -> None:
        """Initialize embeddings, feedback blocks, and classifier.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        d_model:
            Hidden width.
        n_layers:
            Number of repeated feedback blocks per timestep.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([FeedbackBlock(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """Compute token logits using feedback memory.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Logits with shape ``(batch, time, vocab_size)``.
        """
        embedded = self.embed(ids)
        top_states = []
        outputs = []
        for step in range(embedded.shape[1]):
            if top_states:
                memory = torch.stack(top_states, dim=1)
            else:
                memory = embedded.new_zeros(embedded.shape[0], 1, embedded.shape[2])
            state = embedded[:, step]
            for block in self.blocks:
                state = block(state, memory)
            top_states.append(state)
            outputs.append(state)
        return self.head(torch.stack(outputs, dim=1))


def build() -> nn.Module:
    """Build a compact Feedback Transformer.

    Returns
    -------
    nn.Module
        Random-initialized Feedback Transformer language model.
    """
    return FeedbackTransformerLM()


def example_input() -> Tensor:
    """Return example token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 10)``.
    """
    return torch.randint(0, 96, (1, 10), dtype=torch.long)


MENAGERIE_ENTRIES = [("Feedback Transformer", "build", "example_input", "2021", "DC")]
