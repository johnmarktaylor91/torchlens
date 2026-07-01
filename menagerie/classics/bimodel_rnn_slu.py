"""Bi-model RNN for joint intent detection and slot filling.

Paper: A Bi-model based RNN Semantic Frame Parsing Model for Intent Detection and
Slot Filling (Wang, Shen, Jin, NAACL 2018).

The distinctive mechanism is two asynchronous task networks -- one for intent
detection, one for slot filling -- each a BiLSTM over the shared token embedding,
that SHARE each other's hidden-state sequence: the slot network's representation is
conditioned on the intent network's hidden states and vice versa. The two tasks
co-regularize through cross-task hidden-state coupling rather than a single shared
encoder, which is what separates the bi-model from a plain multi-task tagger.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BiModelRNNSLU(nn.Module):
    """Compact Bi-model RNN with cross-task hidden-state sharing."""

    def __init__(
        self,
        vocab_size: int = 64,
        embed_dim: int = 32,
        hidden_dim: int = 32,
        n_intent: int = 8,
        n_slot: int = 16,
    ) -> None:
        """Initialize the embedding, two task BiLSTMs, coupling, and heads."""

        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.intent_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.slot_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Asynchronous cross-task hidden sharing: each task reads the other's hidden sequence.
        self.intent_couple = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
        self.slot_couple = nn.Linear(4 * hidden_dim, 2 * hidden_dim)
        self.intent_head = nn.Linear(2 * hidden_dim, n_intent)
        self.slot_head = nn.Linear(2 * hidden_dim, n_slot)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Return (intent_logits, per-token slot_logits) from shared task hidden states."""

        embedded = self.embed(tokens)
        intent_hidden, _ = self.intent_lstm(embedded)
        slot_hidden, _ = self.slot_lstm(embedded)
        intent_shared = torch.tanh(
            self.intent_couple(torch.cat([intent_hidden, slot_hidden], dim=-1))
        )
        slot_shared = torch.tanh(self.slot_couple(torch.cat([slot_hidden, intent_hidden], dim=-1)))
        intent_logits = self.intent_head(intent_shared.mean(dim=1))
        slot_logits = self.slot_head(slot_shared)
        return intent_logits, slot_logits


def build() -> nn.Module:
    """Build the compact Bi-model RNN SLU."""

    return BiModelRNNSLU().eval()


def example_input() -> Tensor:
    """Return a batch of token ids."""

    return torch.randint(0, 64, (1, 12), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("Bi-model RNN SLU", "build", "example_input", "2018", "SEQ"),
]
