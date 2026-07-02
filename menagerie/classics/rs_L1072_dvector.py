# SOURCE: vendored from https://github.com/yistLin/dvector @ master (modules/dvector.py)
#
# d-vector: DNN-based speaker embedding trained with the GE2E (generalized end-to-end)
# loss (Wan et al. 2018 "Generalized End-to-End Loss for Speaker Verification"). The
# repo's `modules/dvector.py` defines two concrete d-vector encoder architectures --
# `LSTMDvector` (a stacked LSTM over mel-spectrogram frames, last-timestep hidden state
# projected to an embedding and L2-normalized) and `AttentivePooledLSTMDvector` (the
# same stacked LSTM, but every timestep's projected embedding is pooled via a learned
# scalar-attention softmax instead of just taking the last timestep). Both are copied
# verbatim from the repo; only the shared `DvectorInterface.embed_utterance` /
# `embed_utterances` convenience methods (which operate on a single long utterance via
# unfold+average, not part of the traced network graph) are omitted here since they are
# helper wrappers around `forward`, not additional architecture -- the traced entry point
# is the real, unmodified `forward()` of each class. No layer or control-flow logic in
# the traced path was altered.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"


class LSTMDvector(nn.Module):
    """LSTM-based d-vector."""

    def __init__(
        self,
        num_layers=3,
        dim_input=40,
        dim_cell=256,
        dim_emb=256,
        seg_len=160,
    ):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_cell, num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.seg_len = seg_len

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = self.embedding(lstm_outs[:, -1, :])  # (batch, dim_emb)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))  # (batch, dim_emb)


class AttentivePooledLSTMDvector(nn.Module):
    """LSTM-based d-vector with attentive pooling."""

    def __init__(
        self,
        num_layers=3,
        dim_input=40,
        dim_cell=256,
        dim_emb=256,
        seg_len=160,
    ):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_cell, num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.linear = nn.Linear(dim_emb, 1)
        self.seg_len = seg_len

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward a batch through network."""
        lstm_outs, _ = self.lstm(inputs)  # (batch, seg_len, dim_cell)
        embeds = torch.tanh(self.embedding(lstm_outs))  # (batch, seg_len, dim_emb)
        attn_weights = F.softmax(self.linear(embeds), dim=1)
        embeds = torch.sum(embeds * attn_weights, dim=1)
        return embeds.div(embeds.norm(p=2, dim=-1, keepdim=True))


def build_lstm_dvector():
    return LSTMDvector(num_layers=2, dim_input=40, dim_cell=32, dim_emb=32, seg_len=16)


def example_input_lstm_dvector():
    return torch.randn(2, 16, 40)


def build_attentive_pooled_lstm_dvector():
    return AttentivePooledLSTMDvector(
        num_layers=2, dim_input=40, dim_cell=32, dim_emb=32, seg_len=16
    )


def example_input_attentive_pooled_lstm_dvector():
    return torch.randn(2, 16, 40)


MENAGERIE_ENTRIES = [
    ("LSTM d-vector", "build_lstm_dvector", "example_input_lstm_dvector", 2018, MENAGERIE_ZOO),
    (
        "Attentive-Pooled LSTM d-vector",
        "build_attentive_pooled_lstm_dvector",
        "example_input_attentive_pooled_lstm_dvector",
        2018,
        MENAGERIE_ZOO,
    ),
]
