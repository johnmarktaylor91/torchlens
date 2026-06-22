"""AllenNLP BiDAF: bidirectional attention flow for machine comprehension.

Seo et al., 2016, introduced BiDAF as a hierarchical reader with character/word
embedding, contextual encoders, bidirectional context-query attention, modeling
RNN layers, and span start/end predictors. AllenNLP shipped a dependency-gated
PyTorch implementation; this is a compact faithful reconstruction of the same
data flow.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Highway(nn.Module):
    """Highway projection used after token embeddings."""

    def __init__(self, dim: int) -> None:
        """Initialize transform and gate projections.

        Parameters
        ----------
        dim:
            Feature dimension.
        """
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated highway mixing.

        Parameters
        ----------
        x:
            Input token features.

        Returns
        -------
        Tensor
            Highway-transformed features.
        """
        gate = torch.sigmoid(self.gate(x))
        transformed = torch.relu(self.transform(x))
        return gate * transformed + (1.0 - gate) * x


class BiDAFReader(nn.Module):
    """Compact BiDAF span reader."""

    def __init__(self, vocab: int = 512, dim: int = 48, hidden: int = 32) -> None:
        """Initialize embeddings, encoders, attention, and span heads.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        dim:
            Embedding dimension.
        hidden:
            LSTM hidden dimension per direction.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.highway = Highway(dim)
        self.context = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.attn_weight = nn.Linear(6 * hidden, 1)
        self.modeling = nn.LSTM(
            8 * hidden, hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.end_modeling = nn.LSTM(
            2 * hidden, hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.start = nn.Linear(10 * hidden, 1)
        self.end = nn.Linear(10 * hidden, 1)

    def _similarity(self, context: Tensor, query: Tensor) -> Tensor:
        """Compute BiDAF trilinear context-query similarity.

        Parameters
        ----------
        context:
            Context encodings ``(batch, c_len, 2 * hidden)``.
        query:
            Query encodings ``(batch, q_len, 2 * hidden)``.

        Returns
        -------
        Tensor
            Similarity matrix ``(batch, c_len, q_len)``.
        """
        c_len = context.shape[1]
        q_len = query.shape[1]
        c_exp = context.unsqueeze(2).expand(-1, c_len, q_len, -1)
        q_exp = query.unsqueeze(1).expand(-1, c_len, q_len, -1)
        return self.attn_weight(torch.cat([c_exp, q_exp, c_exp * q_exp], dim=-1)).squeeze(-1)

    def forward(self, context_ids: Tensor, query_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Predict start and end logits for answer spans.

        Parameters
        ----------
        context_ids:
            Context token ids with shape ``(batch, c_len)``.
        query_ids:
            Query token ids with shape ``(batch, q_len)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Start and end logits over context positions.
        """
        context_emb = self.highway(self.embed(context_ids))
        query_emb = self.highway(self.embed(query_ids))
        context_enc, _ = self.context(context_emb)
        query_enc, _ = self.context(query_emb)
        sim = self._similarity(context_enc, query_enc)
        c2q = torch.matmul(torch.softmax(sim, dim=-1), query_enc)
        q2c_weights = torch.softmax(sim.max(dim=-1).values, dim=-1).unsqueeze(1)
        q2c = torch.matmul(q2c_weights, context_enc).expand(-1, context_enc.shape[1], -1)
        merged = torch.cat([context_enc, c2q, context_enc * c2q, context_enc * q2c], dim=-1)
        modeled, _ = self.modeling(merged)
        start_logits = self.start(torch.cat([merged, modeled], dim=-1)).squeeze(-1)
        end_modeled, _ = self.end_modeling(modeled)
        end_logits = self.end(torch.cat([merged, end_modeled], dim=-1)).squeeze(-1)
        return start_logits, end_logits


def build() -> nn.Module:
    """Build a compact AllenNLP BiDAF reader.

    Returns
    -------
    nn.Module
        Random-initialized BiDAF model.
    """
    return BiDAFReader()


def example_input() -> tuple[Tensor, Tensor]:
    """Return context and query token ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        Context ids ``(1, 16)`` and query ids ``(1, 7)``.
    """
    return torch.randint(0, 512, (1, 16)), torch.randint(0, 512, (1, 7))


MENAGERIE_ENTRIES = [
    ("bidaf_allennlp", "build", "example_input", "2016", "DE"),
]
