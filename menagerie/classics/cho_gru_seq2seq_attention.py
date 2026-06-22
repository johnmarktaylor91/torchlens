"""Cho GRU encoder-decoder with Bahdanau additive attention.

Cho et al. 2014 introduced gated recurrent encoder-decoder phrase modeling, and
Bahdanau et al. 2015 added differentiable alignment over encoder states. This
compact random-init reconstruction keeps the GRU encoder, teacher-forced GRU
decoder, and additive attention context at each output step.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention module."""

    def __init__(self, hidden: int) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        hidden:
            Hidden-state dimension.
        """
        super().__init__()
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.energy = nn.Linear(hidden, 1, bias=False)

    def forward(self, query: Tensor, values: Tensor) -> Tensor:
        """Compute a context vector over encoder states.

        Parameters
        ----------
        query:
            Decoder state with shape ``(batch, hidden)``.
        values:
            Encoder states with shape ``(batch, time, hidden)``.

        Returns
        -------
        Tensor
            Attention context vector.
        """
        scores = self.energy(torch.tanh(self.key(values) + self.query(query).unsqueeze(1)))
        weights = torch.softmax(scores.squeeze(-1), dim=-1)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)


class ChoGRUSeq2SeqAttention(nn.Module):
    """Compact GRU encoder-decoder with Bahdanau attention."""

    def __init__(
        self, vocab: int = 48, dim: int = 32, hidden: int = 48, out_steps: int = 6
    ) -> None:
        """Initialize embeddings, GRUs, attention, and output projection.

        Parameters
        ----------
        vocab:
            Source and target vocabulary size.
        dim:
            Embedding dimension.
        hidden:
            GRU hidden-state size.
        out_steps:
            Number of teacher-forced decoder steps.
        """
        super().__init__()
        self.out_steps = out_steps
        self.src_embed = nn.Embedding(vocab, dim)
        self.tgt_embed = nn.Embedding(vocab, dim)
        self.encoder = nn.GRU(dim, hidden, batch_first=True)
        self.decoder = nn.GRUCell(dim + hidden, hidden)
        self.attn = AdditiveAttention(hidden)
        self.output = nn.Linear(hidden * 2, vocab)

    def forward(self, src: Tensor, tgt: Tensor | None = None) -> Tensor:
        """Translate source tokens with teacher-forced attention decoding.

        Parameters
        ----------
        src:
            Source token ids with shape ``(batch, source_time)``.
        tgt:
            Optional target token ids with shape ``(batch, target_time)``.

        Returns
        -------
        Tensor
            Target vocabulary logits.
        """
        if tgt is None:
            tgt = torch.zeros(src.shape[0], self.out_steps, dtype=torch.long, device=src.device)
        encoded, state = self.encoder(self.src_embed(src))
        hidden = state.squeeze(0)
        outputs: list[Tensor] = []
        embedded = self.tgt_embed(tgt)
        for step in range(tgt.shape[1]):
            context = self.attn(hidden, encoded)
            hidden = self.decoder(torch.cat((embedded[:, step], context), dim=-1), hidden)
            outputs.append(self.output(torch.cat((hidden, context), dim=-1)))
        return torch.stack(outputs, dim=1)


class LSTMSeq2SeqBahdanau(nn.Module):
    """Compact LSTM encoder-decoder with Bahdanau additive attention."""

    def __init__(
        self, vocab: int = 48, dim: int = 32, hidden: int = 48, out_steps: int = 6
    ) -> None:
        """Initialize embeddings, LSTM cells, attention, and output projection.

        Parameters
        ----------
        vocab:
            Source and target vocabulary size.
        dim:
            Embedding dimension.
        hidden:
            LSTM hidden-state size.
        out_steps:
            Number of decoder steps when no target is supplied.
        """
        super().__init__()
        self.out_steps = out_steps
        self.src_embed = nn.Embedding(vocab, dim)
        self.tgt_embed = nn.Embedding(vocab, dim)
        self.encoder = nn.LSTM(dim, hidden, batch_first=True)
        self.decoder = nn.LSTMCell(dim + hidden, hidden)
        self.attn = AdditiveAttention(hidden)
        self.output = nn.Linear(hidden * 2, vocab)

    def forward(self, src: Tensor, tgt: Tensor | None = None) -> Tensor:
        """Translate source tokens with LSTM attention decoding.

        Parameters
        ----------
        src:
            Source token ids with shape ``(batch, source_time)``.
        tgt:
            Optional target token ids with shape ``(batch, target_time)``.

        Returns
        -------
        Tensor
            Target vocabulary logits.
        """
        if tgt is None:
            tgt = torch.zeros(src.shape[0], self.out_steps, dtype=torch.long, device=src.device)
        encoded, (hidden_state, cell_state) = self.encoder(self.src_embed(src))
        hidden = hidden_state.squeeze(0)
        cell = cell_state.squeeze(0)
        embedded = self.tgt_embed(tgt)
        outputs: list[Tensor] = []
        for step in range(tgt.shape[1]):
            context = self.attn(hidden, encoded)
            hidden, cell = self.decoder(
                torch.cat((embedded[:, step], context), dim=-1), (hidden, cell)
            )
            outputs.append(self.output(torch.cat((hidden, context), dim=-1)))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact random-init GRU seq2seq attention model.

    Returns
    -------
    nn.Module
        Compact seq2seq model.
    """
    return ChoGRUSeq2SeqAttention()


def build_lstm_bahdanau() -> nn.Module:
    """Build a compact LSTM seq2seq model with Bahdanau attention.

    Returns
    -------
    nn.Module
        Compact LSTM attention model.
    """
    return LSTMSeq2SeqBahdanau()


def example_input() -> Tensor:
    """Return source token ids.

    Returns
    -------
    Tensor
        Source tokens.
    """
    return torch.randint(0, 48, (1, 8))


MENAGERIE_ENTRIES = [
    ("ChoGRUEncoderDecoder", "build", "example_input", "2014", "E7"),
    ("Seq2SeqLSTM-Attention-Bahdanau", "build_lstm_bahdanau", "example_input", "2015", "E7"),
]
