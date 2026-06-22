"""TransAct: Transformer-based Real-Time User Action Model for Recommendation.

Tang et al., "TransAct: Transformer-based Realtime User Action Model for Twitter
Recommendation", KDD 2023. arXiv:2306.00248
Source: https://github.com/twitter/the-algorithm-ml (Pinterest/Twitter research)
FuxiCTR reimplementation: https://github.com/xue-pai/FuxiCTR

TransAct models a user's recent action sequence via a Transformer encoder,
then concatenates the sequence summary with other (non-sequential) features
and passes through a DNN for CTR prediction.

Architecture:
  Sequential features (user recent actions):
    item_id sequence -> Embedding -> positional encoding ->
    Transformer encoder (self-attention over the sequence) ->
    target-item cross-attention or pooling -> sequence_repr

  Other features:
    (B, num_other_fields) -> Embedding -> flatten -> other_repr

  Concat [sequence_repr, other_repr] -> DNN -> logit

Faithful-core simplifications:
  - Sequence length reduced to 8 (paper uses ~100+)
  - Transformer: 2 layers, 2 heads
  - DNN: 2 hidden layers [64, 32]
  - Item vocab: 200 items; other feature vocab: 50
  - Random init, CPU, trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 8
NUM_OTHER_FIELDS = 6
EMBED_DIM = 8
VOCAB = 200
VOCAB_OTHER = 50
BATCH = 2


class TransActModel(nn.Module):
    """TransAct: Transformer over user action sequence + DNN.

    Forward inputs:
      x_seq:   (B, seq_len) integer item IDs (recent actions)
      x_other: (B, num_other_fields) integer categorical features
    """

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        num_other_fields: int = NUM_OTHER_FIELDS,
        embed_dim: int = EMBED_DIM,
        vocab: int = VOCAB,
        vocab_other: int = VOCAB_OTHER,
        num_transformer_layers: int = 2,
        num_heads: int = 2,
        dnn_hidden: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dnn_hidden is None:
            dnn_hidden = [64, 32]

        self.seq_embedding = nn.Embedding(vocab, embed_dim)
        self.pos_enc = nn.Embedding(seq_len, embed_dim)  # learned positional
        self.other_embedding = nn.Embedding(vocab_other, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Aggregate sequence -> single vector (mean pooling)
        # DNN input: seq_repr + other_repr
        dnn_in = embed_dim + num_other_fields * embed_dim
        layers: List[nn.Module] = []
        prev = dnn_in
        for h in dnn_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x_seq: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, S), x_other: (B, F)
        B, S = x_seq.shape
        pos = torch.arange(S, device=x_seq.device).unsqueeze(0)  # (1, S)
        seq_emb = self.seq_embedding(x_seq.long()) + self.pos_enc(pos)  # (B, S, E)

        # Transformer over action sequence
        seq_out = self.transformer(seq_emb)  # (B, S, E)
        seq_repr = seq_out.mean(dim=1)  # (B, E) -- mean pooling

        # Other features
        other_emb = self.other_embedding(x_other.long())  # (B, F, E)
        other_repr = other_emb.flatten(1)  # (B, F*E)

        # Concat and DNN
        h = torch.cat([seq_repr, other_repr], dim=-1)  # (B, E + F*E)
        return self.dnn(h).squeeze(-1)  # (B,)


def build() -> nn.Module:
    """TransAct: Transformer over user action sequence + context features -> CTR."""
    return TransActModel()


def example_input() -> List[torch.Tensor]:
    """Returns [x_seq, x_other] for TransAct."""
    x_seq = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    x_other = torch.randint(0, VOCAB_OTHER, (BATCH, NUM_OTHER_FIELDS))
    return [x_seq, x_other]


MENAGERIE_ENTRIES = [
    (
        "TransAct (Transformer over user action sequence + features -> CTR)",
        "build",
        "example_input",
        "2023",
        "DC",
    ),
]
