"""Adaptive input representations for neural language modeling.

Paper: "Adaptive Input Representations for Neural Language Modeling",
Baevski and Auli, ICLR 2019.

The architecture factorizes token input embeddings by frequency bucket: frequent
words receive full-width embeddings, while progressively rarer buckets use lower
dimensional embeddings projected back to the model width. This compact language
model keeps the bucketed input factorization and a small recurrent LM head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdaptiveInputEmbedding(nn.Module):
    """Frequency-bucketed adaptive input embedding layer."""

    def __init__(
        self,
        cutoffs: tuple[int, ...] = (32, 96, 160),
        dims: tuple[int, ...] = (32, 16, 8),
        d_model: int = 32,
    ) -> None:
        """Initialize bucket embeddings and projections.

        Parameters
        ----------
        cutoffs:
            Exclusive vocabulary cutoffs for each bucket.
        dims:
            Embedding width for each bucket.
        d_model:
            Shared output width after projection.
        """
        super().__init__()
        starts = (0,) + cutoffs[:-1]
        self.starts = starts
        self.ends = cutoffs
        self.embeddings = nn.ModuleList(
            [nn.Embedding(end - start, dim) for start, end, dim in zip(starts, cutoffs, dims)]
        )
        self.projections = nn.ModuleList([nn.Linear(dim, d_model, bias=False) for dim in dims])

    def forward(self, ids: Tensor) -> Tensor:
        """Embed token ids with bucket-specific capacity.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Projected embeddings with shape ``(batch, time, d_model)``.
        """
        ids = ids.clamp(min=0, max=self.ends[-1] - 1)
        first = self.projections[0](self.embeddings[0](ids.clamp(max=self.ends[0] - 1)))
        out = torch.zeros_like(first)
        for start, end, emb, proj in zip(self.starts, self.ends, self.embeddings, self.projections):
            rel_ids = (ids - start).clamp(min=0, max=end - start - 1)
            mask = ((ids >= start) & (ids < end)).unsqueeze(-1).to(first.dtype)
            out = out + proj(emb(rel_ids)) * mask
        return out


class AdaptiveInputLM(nn.Module):
    """Small language model with adaptive input embeddings."""

    def __init__(self, vocab_size: int = 160, d_model: int = 32) -> None:
        """Initialize adaptive inputs, recurrent body, and classifier.

        Parameters
        ----------
        vocab_size:
            Vocabulary size represented by the adaptive buckets.
        d_model:
            Shared hidden width.
        """
        super().__init__()
        self.inputs = AdaptiveInputEmbedding(d_model=d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """Compute token logits from token ids.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Language-model logits with shape ``(batch, time, vocab_size)``.
        """
        hidden, _ = self.rnn(self.inputs(ids))
        return self.classifier(self.norm(hidden))


def build() -> nn.Module:
    """Build a compact adaptive-input language model.

    Returns
    -------
    nn.Module
        Random-initialized adaptive-input model.
    """
    return AdaptiveInputLM()


def example_input() -> Tensor:
    """Return example token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 16)``.
    """
    return torch.randint(0, 160, (1, 16), dtype=torch.long)


MENAGERIE_ENTRIES = [
    (
        "Adaptive Input Representations",
        "build",
        "example_input",
        "2019",
        "DE",
    )
]
