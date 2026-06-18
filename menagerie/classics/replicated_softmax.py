"""Replicated Softmax Model, 2009, Salakhutdinov and Hinton.

Paper: Replicated Softmax: an Undirected Topic Model.
An RBM topic model for word-count vectors where hidden-topic biases are driven
by counts and visible logits are scaled by document length.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReplicatedSoftmax(nn.Module):
    """RBM-style topic model for bag-of-words counts."""

    def __init__(self, vocab_size: int = 20, n_topics: int = 6) -> None:
        """Initialize replicated softmax parameters.

        Parameters
        ----------
        vocab_size:
            Number of vocabulary entries.
        n_topics:
            Number of hidden topic units.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, n_topics) * 0.04)
        self.word_bias = nn.Parameter(torch.zeros(vocab_size))
        self.topic_bias = nn.Parameter(torch.zeros(n_topics))

    def forward(self, counts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute hidden topic probabilities and reconstruction logits.

        Parameters
        ----------
        counts:
            Word-count matrix of shape ``(batch, vocab_size)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Topic probabilities, word probabilities, and document lengths.
        """
        lengths = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        topic_logits = counts @ self.weight + lengths * self.topic_bias
        topic_probs = torch.sigmoid(topic_logits)
        word_logits = self.word_bias + topic_probs @ self.weight.T
        word_probs = torch.softmax(word_logits, dim=-1)
        return topic_probs, word_probs, lengths


def build() -> nn.Module:
    """Build a small replicated softmax model.

    Returns
    -------
    nn.Module
        ReplicatedSoftmax instance.
    """
    return ReplicatedSoftmax()


def example_input() -> Tensor:
    """Return a sample word-count batch.

    Returns
    -------
    Tensor
        Float count tensor of shape ``(2, 20)``.
    """
    return torch.poisson(torch.full((2, 20), 1.5))
