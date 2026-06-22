"""Single Headed Attention RNN (SHA-RNN).

Paper: "Single Headed Attention RNN: Stop Thinking With Your Head",
Merity, arXiv 2019.

SHA-RNN keeps an LSTM recurrent core but adds a bounded single-head attention
module, layer normalization, and a BOOM feed-forward block for language modeling.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Boom(nn.Module):
    """BOOM feed-forward block used by SHA-RNN."""

    def __init__(self, width: int, expansion: int = 4) -> None:
        """Initialize expand/project layers.

        Parameters
        ----------
        width:
            Input and output width.
        expansion:
            Expansion factor for the hidden feed-forward layer.
        """
        super().__init__()
        self.up = nn.Linear(width, expansion * width)
        self.down = nn.Linear(expansion * width, width)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the BOOM block.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Feed-forward output with the same shape as ``x``.
        """
        return self.down(torch.relu(self.up(x)))


class SingleHeadAttention(nn.Module):
    """Causal single-head attention over recurrent states."""

    def __init__(self, width: int) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        width:
            Hidden width.
        """
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal single-head self-attention.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, width)``.

        Returns
        -------
        Tensor
            Attention output with the same shape as ``x``.
        """
        scores = torch.matmul(self.q(x), self.k(x).transpose(1, 2)) / math.sqrt(x.shape[-1])
        time = x.shape[1]
        mask = torch.triu(torch.ones(time, time, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0) > 0, -1.0e4)
        return self.out(torch.matmul(torch.softmax(scores, dim=-1), self.v(x)))


class SHARNN(nn.Module):
    """Compact SHA-RNN language model."""

    def __init__(self, vocab_size: int = 96, width: int = 32) -> None:
        """Initialize embedding, LSTM, attention, BOOM block, and classifier.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        width:
            Embedding and recurrent hidden width.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.rnn = nn.LSTM(width, width, batch_first=True)
        self.norm_rnn = nn.LayerNorm(width)
        self.attn = SingleHeadAttention(width)
        self.norm_attn = nn.LayerNorm(width)
        self.boom = Boom(width)
        self.norm_boom = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """Compute token logits from ids.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Logits with shape ``(batch, time, vocab_size)``.
        """
        seq, _ = self.rnn(self.embed(ids))
        seq = self.norm_rnn(seq)
        seq = self.norm_attn(seq + self.attn(seq))
        seq = self.norm_boom(seq + self.boom(seq))
        return self.head(seq)


def build() -> nn.Module:
    """Build a compact SHA-RNN.

    Returns
    -------
    nn.Module
        Random-initialized SHA-RNN model.
    """
    return SHARNN()


def example_input() -> Tensor:
    """Return example token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 12)``.
    """
    return torch.randint(0, 96, (1, 12), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("SHA-RNN (Single Headed Attention RNN)", "build", "example_input", "2019", "DE")
]
