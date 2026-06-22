"""Pointer Sentinel-LSTM mixture model.

Paper: "Pointer Sentinel Mixture Models", Merity et al., ICLR 2017.

Pointer Sentinel-LSTM augments a recurrent language model with a pointer cache
over recent hidden states and a sentinel that lets the pointer component choose
between copying from history and using the normal vocabulary softmax.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PointerSentinelLSTM(nn.Module):
    """Compact pointer-sentinel recurrent language model."""

    def __init__(self, vocab_size: int = 96, width: int = 32) -> None:
        """Initialize embedding, LSTM, vocabulary head, and sentinel gate.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        width:
            Embedding and recurrent hidden width.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, width)
        self.rnn = nn.LSTM(width, width, batch_first=True)
        self.vocab_head = nn.Linear(width, vocab_size)
        self.sentinel = nn.Linear(width, 1)

    def forward(self, ids: Tensor) -> Tensor:
        """Compute vocabulary probabilities mixed with a pointer cache.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Mixed probabilities with shape ``(batch, time, vocab_size)``.
        """
        ids = ids.clamp(min=0, max=self.vocab_size - 1)
        hidden, _ = self.rnn(self.embed(ids))
        vocab_probs = torch.softmax(self.vocab_head(hidden), dim=-1)

        scores = torch.matmul(hidden, hidden.transpose(1, 2)) / math.sqrt(hidden.shape[-1])
        time = ids.shape[1]
        causal = torch.tril(torch.ones(time, time, device=ids.device), diagonal=-1)
        scores = scores.masked_fill(causal.unsqueeze(0) == 0, -1.0e4)
        sentinel_scores = self.sentinel(hidden)
        pointer_with_sentinel = torch.softmax(torch.cat([scores, sentinel_scores], dim=-1), dim=-1)
        pointer_weights = pointer_with_sentinel[:, :, :time]
        sentinel_weight = pointer_with_sentinel[:, :, time:].clamp(0.0, 1.0)

        pointer_vocab = torch.zeros_like(vocab_probs)
        scatter_index = ids.unsqueeze(1).expand(-1, time, -1)
        pointer_vocab = pointer_vocab.scatter_add(-1, scatter_index, pointer_weights)
        return sentinel_weight * vocab_probs + pointer_vocab


def build() -> nn.Module:
    """Build a compact Pointer Sentinel-LSTM.

    Returns
    -------
    nn.Module
        Random-initialized Pointer Sentinel-LSTM model.
    """
    return PointerSentinelLSTM()


def example_input() -> Tensor:
    """Return example token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 12)``.
    """
    return torch.randint(0, 96, (1, 12), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("Pointer Sentinel-LSTM mixture model", "build", "example_input", "2017", "DE")
]
