# SOURCE: vendored from ray075hl/Bi-Model-Intent-And-Slot @ master (model.py)
"""Bi-model RNN for joint intent detection and slot filling.

Paper: Wang, Shen & Jin, "A Bi-Model Based RNN Semantic Frame Parsing Model
for Intent Detection and Slot Filling" (NAACL 2018).

The four encoder/decoder classes below (``slot_enc``, ``slot_dec``,
``intent_enc``, ``intent_dec``) are copied verbatim from the official repo's
``model.py``. Only the constructor default that reads vocab/label sizes from
on-disk dictionaries (``make_dict.word_dict`` / ``intent_dict`` /
``slot_dict``, built from the ATIS corpus files) and the hard-coded
``device``/``config`` module globals were removed so the classes can be
constructed directly with small explicit sizes -- no architecture was
changed. The bidirectional forward wiring (each tower's encoder output is
fed as ``share_memory`` into the other tower's decoder) mirrors
``train.py``'s ``slot_model.dec(hs, intent_model.share_memory.detach())`` /
``intent_model.dec(hi, slot_model.share_memory.detach(), real_len)`` calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.2


class slot_enc(nn.Module):
    """Bidirectional 2-layer LSTM encoder for the slot-filling tower."""

    def __init__(self, embedding_size: int, lstm_hidden_size: int, vocab_size: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = F.dropout(x, DROPOUT)
        x, _ = self.lstm(x)
        x = F.dropout(x, DROPOUT)
        return x


class slot_dec(nn.Module):
    """Autoregressive slot-label decoder fed by the intent tower's share_memory."""

    def __init__(self, lstm_hidden_size: int, label_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size * 5, hidden_size=lstm_hidden_size, num_layers=1
        )
        self.fc = nn.Linear(lstm_hidden_size, label_size)
        self.hidden_size = lstm_hidden_size

    def forward(self, x: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        length = x.size(1)
        dec_init_out = torch.zeros(batch, 1, self.hidden_size)
        hidden_state = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )
        x = torch.cat((x, hi), dim=-1)

        x = x.transpose(1, 0)  # length x batch x feature_size
        x = F.dropout(x, DROPOUT)
        all_out = []
        for i in range(length):
            if i == 0:
                out, hidden_state = self.lstm(
                    torch.cat((x[i].unsqueeze(1), dec_init_out), dim=-1), hidden_state
                )
            else:
                out, hidden_state = self.lstm(
                    torch.cat((x[i].unsqueeze(1), out), dim=-1), hidden_state
                )
            all_out.append(out)
        output = torch.cat(all_out, dim=1)  # batch x length x feature_size
        x = F.dropout(x, DROPOUT)
        res = self.fc(output)
        return res


class intent_enc(nn.Module):
    """Bidirectional 2-layer LSTM encoder for the intent-detection tower."""

    def __init__(self, embedding_size: int, lstm_hidden_size: int, vocab_size: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = F.dropout(x, DROPOUT)
        x, _ = self.lstm(x)
        x = F.dropout(x, DROPOUT)
        return x


class intent_dec(nn.Module):
    """Intent-label decoder fed by the slot tower's share_memory."""

    def __init__(self, lstm_hidden_size: int, label_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size * 4,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.fc = nn.Linear(lstm_hidden_size, label_size)

    def forward(self, x: torch.Tensor, hs: torch.Tensor, real_len: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, hs), dim=-1)
        x = F.dropout(x, DROPOUT)
        x, _ = self.lstm(x)
        x = F.dropout(x, DROPOUT)

        batch = x.size()[0]
        index = torch.arange(batch).long()
        state = x[index, real_len - 1, :]

        res = self.fc(state.squeeze(1) if state.dim() > 2 else state)
        return res


class BiModelSLU(nn.Module):
    """Joint Bi-model wrapper wiring the two encoder/decoder towers together.

    Reproduces the cross-tower ``share_memory`` feed used in the official
    ``train.py`` training loop as a single ``forward`` for TorchLens capture.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        embedding_size: int = 16,
        lstm_hidden_size: int = 12,
        slot_label_size: int = 10,
        intent_label_size: int = 6,
    ) -> None:
        super().__init__()
        self.slot_enc = slot_enc(embedding_size, lstm_hidden_size, vocab_size)
        self.slot_dec = slot_dec(lstm_hidden_size, slot_label_size)
        self.intent_enc = intent_enc(embedding_size, lstm_hidden_size, vocab_size)
        self.intent_dec = intent_dec(lstm_hidden_size, intent_label_size)

    def forward(self, x: torch.Tensor, real_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hs = self.slot_enc(x)
        hi = self.intent_enc(x)

        slot_logits = self.slot_dec(hs, hi.detach())
        intent_logits = self.intent_dec(hi, hs.detach(), real_len)
        return slot_logits, intent_logits


def build_bimodel_slu() -> nn.Module:
    """Build a small Bi-model joint intent/slot SLU network.

    Returns
    -------
    nn.Module
        Random-initialized ``BiModelSLU``.
    """

    return BiModelSLU()


def example_input_bimodel_slu() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a token-id batch and per-example sequence length.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(token_ids, real_len)`` with shapes ``(2, 8)`` and ``(2,)``.
    """

    x = torch.randint(0, 64, (2, 8))
    real_len = torch.tensor([8, 8])
    return x, real_len


MENAGERIE_ENTRIES = [
    ("Bi-model RNN SLU", "build_bimodel_slu", "example_input_bimodel_slu", "2018", "nlp_slu"),
]

MENAGERIE_ZOO = "vendored-pytorch"
