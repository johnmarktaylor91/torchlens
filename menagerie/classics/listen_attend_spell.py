"""Listen-Attend-Spell, 2015, Chan et al., "Listen, Attend and Spell".

A pyramidal listener compresses acoustic frames and a recurrent speller emits
characters with additive attention. This minimal teacher-forced version omits
beam search and scheduled sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ListenAttendSpell(nn.Module):
    """Small LAS encoder-decoder with Bahdanau attention."""

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 32,
        vocab_size: int = 30,
    ) -> None:
        """Initialize listener, speller, and attention projections.

        Parameters
        ----------
        input_size:
            Acoustic feature size.
        hidden_size:
            Recurrent state size.
        vocab_size:
            Character vocabulary size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.listener_in = nn.Linear(input_size, 2 * hidden_size)
        self.pyramid_layers = nn.ModuleList(
            [
                nn.LSTM(4 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
                for _ in range(3)
            ]
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.decoder = nn.LSTMCell(3 * hidden_size, hidden_size)
        self.key = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy = nn.Linear(hidden_size, 1, bias=False)
        self.classifier = nn.Linear(3 * hidden_size, vocab_size)

    def _halve_time(self, seq: Tensor) -> Tensor:
        """Stack adjacent frames to halve the time axis.

        Parameters
        ----------
        seq:
            Sequence tensor with shape ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Time-halved tensor with doubled feature size.
        """
        batch, steps, features = seq.shape
        even_steps = steps - (steps % 2)
        seq = seq[:, :even_steps]
        return seq.reshape(batch, even_steps // 2, features * 2)

    def forward(self, feats: Tensor, chars: Tensor | None = None) -> Tensor:
        """Decode characters from acoustic features with teacher forcing.

        Parameters
        ----------
        feats:
            Acoustic tensor with shape ``(batch, time, input_size)``.
        chars:
            Optional teacher character ids with shape ``(batch, out_time)``. If
            omitted, a zero prompt is used for trace-friendly inference.

        Returns
        -------
        Tensor
            Character logits with shape ``(batch, out_time, vocab_size)``.
        """
        if chars is None:
            chars = torch.zeros(feats.shape[0], 12, dtype=torch.long, device=feats.device)
        encoded = torch.tanh(self.listener_in(feats))
        for layer in self.pyramid_layers:
            encoded, _ = layer(self._halve_time(encoded))
        keys = self.key(encoded)
        state = feats.new_zeros(feats.shape[0], self.hidden_size)
        cell = feats.new_zeros(feats.shape[0], self.hidden_size)
        context = feats.new_zeros(feats.shape[0], 2 * self.hidden_size)
        outputs: list[Tensor] = []
        embeddings = self.embed(chars)
        for step in range(chars.shape[1]):
            state, cell = self.decoder(
                torch.cat((embeddings[:, step], context), dim=-1), (state, cell)
            )
            scores = self.energy(torch.tanh(keys + self.query(state).unsqueeze(1))).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
            outputs.append(self.classifier(torch.cat((state, context), dim=-1)))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact Listen-Attend-Spell model.

    Returns
    -------
    nn.Module
        Random-initialized LAS model.
    """
    return ListenAttendSpell()


def example_input() -> Tensor:
    """Return example acoustic features.

    Returns
    -------
    Tensor
        Feature tensor with shape ``(1, 400, 80)``.
    """
    return torch.randn(1, 400, 80)


MENAGERIE_ENTRIES = [("Listen-Attend-Spell (LAS)", "build", "example_input", "2015", "DE")]
