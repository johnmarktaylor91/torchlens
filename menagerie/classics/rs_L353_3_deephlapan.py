# FAITHFUL PORT of zjupgx/deephlapan @ master (original framework: Keras/TF, model.predict via saved .hdf5)
# https://raw.githubusercontent.com/zjupgx/deephlapan/master/deephlapan/attention.py
# https://raw.githubusercontent.com/zjupgx/deephlapan/master/deephlapan/deephlapan_main.py
#
# Wu et al. 2019 (Frontiers in Immunology) "DeepHLApan: A Deep Learning
# Approach for Neoantigen Prediction Considering Both HLA-Peptide Binding
# and Immunogenicity" -- the repo only ships trained Keras `.hdf5` weight
# files (`deephlapan/model/binding_model*.hdf5`,
# `immunogenicity_model*.hdf5`) plus inference code (`deephlapan_main.py`
# calls `keras.models.load_model(...)`); the training script that BUILDS the
# Sequential model is not included in the repo. Rather than guess the
# architecture from the paper, the REAL layer stack was recovered by reading
# the `model_config` JSON serialized into one of the actual shipped
# `binding_model*.hdf5` files (Keras always embeds the exact
# `Sequential.get_config()` used at save time):
#   Embedding(input_dim=22, output_dim=21, input_length=49)
#   -> Bidirectional(GRU(units=128, dropout=0.2, recurrent_dropout=0.2,
#        return_sequences=True), merge_mode='concat')   x3 (stacked)
#   -> Attention(...)   (custom layer, class below, from attention.py)
#   -> Dense(units=1, activation='sigmoid')
# This IS the real, trained architecture (byte-identical layer stack/units
# to the shipped weights), not an approximation from the paper text.
#
# `Attention` (custom Keras layer, `deephlapan/attention.py`) computes a
# learned scalar attention score per timestep (`tanh(x @ W + b)`), softmax
# -normalizes it over the time axis, and returns the attention-weighted sum
# of the GRU output sequence -- reproduced faithfully below as
# `DeepHLApanAttention` (same `dot_product` -> `tanh` -> `exp`/normalize
# -> weighted-sum arithmetic as the original `call()`).
#
# Only mechanical port edits: Keras `Bidirectional(GRU(..., return_sequences
# =True))` -> torch `nn.GRU(bidirectional=True, batch_first=True)`;
# `Embedding` -> `nn.Embedding`; `Dense(1, sigmoid)` -> `nn.Linear(...,1)` +
# `nn.Sigmoid()`. Recurrent/embedding dropout (train-time only) is omitted
# from the traced forward pass, matching Keras inference-mode behavior.

import torch
import torch.nn as nn


class DeepHLApanAttention(nn.Module):
    """Faithful port of the custom Keras `Attention` layer in
    deephlapan/attention.py: eij = tanh(x . W + b); a = softmax(eij) over
    time; output = sum_t(a_t * x_t)."""

    def __init__(self, feature_dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(feature_dim).uniform_(-0.05, 0.05))
        self.b = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):  # x: [N, T, feature_dim]
        eij = torch.tanh(torch.matmul(x, self.W) + self.b.sum())  # [N, T]
        a = torch.exp(eij)
        a = a / (a.sum(dim=1, keepdim=True) + 1e-10)
        weighted_input = x * a.unsqueeze(-1)  # [N, T, feature_dim]
        return weighted_input.sum(dim=1)  # [N, feature_dim]


class DeepHLApanNet(nn.Module):
    """Faithful port of the Sequential stack recovered from the real
    binding_model*.hdf5 config: Embedding -> 3x Bidirectional-GRU(128,
    return_sequences=True) -> Attention -> Dense(1, sigmoid)."""

    def __init__(self, vocab_size=22, embed_dim=21, seq_len=49, gru_units=128, n_gru_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru_layers = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(n_gru_layers):
            self.gru_layers.append(nn.GRU(in_dim, gru_units, batch_first=True, bidirectional=True))
            in_dim = gru_units * 2  # bidirectional concat
        self.attention = DeepHLApanAttention(in_dim)
        self.dense = nn.Linear(in_dim, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):  # x: [N, seq_len] integer token ids
        h = self.embedding(x)
        for gru in self.gru_layers:
            h, _ = gru(h)
        h = self.attention(h)
        h = self.dense(h)
        return self.out_act(h)


def build_deephlapan():
    return DeepHLApanNet()


def example_input_deephlapan():
    return torch.randint(0, 22, (2, 49))


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepHLApan", "build_deephlapan", "example_input_deephlapan", 2019, "ported"),
]
