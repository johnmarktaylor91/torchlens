# FAITHFUL PORT of https://github.com/LogIntelligence/NeuralLog @ main
# (original framework: TensorFlow/Keras)
# (neurallog/models/transformers.py::TransformerBlock/transformer_classifer +
#  neurallog/models/positional_encodings.py::positional_encoding/PositionEmbedding)
#
# NeuralLog (Le & Zhang, ASE 2021, "Log-based Anomaly Detection Without Log
# Parsing"): a Transformer-encoder log-anomaly classifier that consumes a sequence
# of pretrained semantic log-line embeddings (BERT/RoBERTa/etc. sentence embeddings
# computed offline by neurallog/data_loader.py, NOT part of this architecture) and
# classifies the sequence as normal/anomalous. Architecture: sinusoidal absolute
# position embedding added to the input embedding sequence -> a single
# Transformer encoder block (multi-head self-attention + 2-layer FFN, each with a
# residual + LayerNorm, "post-norm" as in the original Transformer) -> global
# average pooling over the sequence -> a small MLP classifier head with two
# Dropout layers -> 2-way softmax (normal/anomalous).
#
# The real repo is TensorFlow/Keras (tensorflow.keras.layers.MultiHeadAttention /
# Dense / LayerNormalization / Dropout / GlobalAveragePooling1D, functional-API
# `keras.Model`), not installed in this base torch env and not vendorable as-is.
# This is a faithful, mechanism-for-mechanism torch transcription of
# `TransformerBlock.call` and `transformer_classifer` (transformers.py) plus
# `PositionEmbedding.call`/`positional_encoding` (positional_encodings.py):
#   - `layers.MultiHeadAttention(num_heads, key_dim=embed_dim)` (self-attention,
#     Q=K=V=inputs) -> `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)`
#     called as self-attention (query=key=value=x).
#   - `keras.Sequential([Dense(ff_dim, relu), Dense(embed_dim)])` -> the identical
#     2-layer `Linear -> ReLU -> Linear` FFN.
#   - `LayerNormalization(epsilon=1e-6)` x2, applied post-residual exactly as in
#     the original `call()` (`layernorm1(inputs + attn_output)`, then
#     `layernorm2(out1 + ffn_output)`) -> `nn.LayerNorm(embed_dim, eps=1e-6)` x2,
#     same residual placement.
#   - `Dropout(rate)` x2 inside the block, at the same two points (post-attention,
#     post-FFN).
#   - `positional_encoding(position, d_model)`: the identical
#     sin(pos/10000^(2i/d))/cos(...) formula, precomputed as a fixed (non-learned)
#     buffer, added to the input exactly as `x += self.pos_encoding[:, :seq_len, :]`.
#   - `transformer_classifer`: PositionEmbedding -> TransformerBlock ->
#     GlobalAveragePooling1D (`x.mean(dim=1)`) -> Dropout -> Dense(32, relu) ->
#     Dropout -> Dense(2, softmax), same layer order/widths/activations.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


# --------------------------------------------------------------------------------
# neurallog/models/positional_encodings.py
# --------------------------------------------------------------------------------


def _positional_encoding(position: int, d_model: int) -> torch.Tensor:
    """Faithful port of `positional_encoding` (numpy sin/cos formula), returned
    as a torch tensor of shape (1, position, d_model)."""
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)  # (position, 1)
    i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)  # (1, d_model)
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates  # (position, d_model)

    pos_encoding = angle_rads.clone()
    pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    return pos_encoding.unsqueeze(0)  # (1, position, d_model)


class PositionEmbedding(nn.Module):
    """Faithful port of `PositionEmbedding(layers.Layer)`: adds a fixed sinusoidal
    positional encoding (precomputed for `max_len` positions) to the input."""

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.register_buffer("pos_encoding", _positional_encoding(max_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        return x


# --------------------------------------------------------------------------------
# neurallog/models/transformers.py
# --------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Faithful port of `TransformerBlock(layers.Layer)`: self multi-head
    attention -> residual + LayerNorm -> 2-layer FFN -> residual + LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class NeuralLog(nn.Module):
    """Faithful port of `transformer_classifer(embed_dim, ff_dim, max_len,
    num_heads, dropout)`: PositionEmbedding -> TransformerBlock -> global average
    pool over the sequence axis -> Dropout -> Dense(32, relu) -> Dropout ->
    Dense(2, softmax)."""

    def __init__(
        self, embed_dim: int, ff_dim: int, max_len: int, num_heads: int, dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_layer = PositionEmbedding(1024, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(embed_dim, 32)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = x.mean(dim=1)  # GlobalAveragePooling1D
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        outputs = self.softmax(x)
        return outputs


def build_neurallog():
    # Real repo demo defaults (demo/NeuralLog.py, examples/*.py): embed_dim=768
    # (BERT-sized sentence embeddings), ff_dim=2048, max_len=75, num_heads=12;
    # shrunk here for a fast, small trace while keeping num_heads a divisor of
    # embed_dim as nn.MultiheadAttention requires.
    return NeuralLog(embed_dim=16, ff_dim=32, max_len=20, num_heads=2, dropout=0.1)


def example_input_neurallog():
    # (batch, seq_len<=max_len, embed_dim): a sequence of pretrained log-line
    # semantic embeddings (BERT-style sentence vectors in the real pipeline).
    return torch.randn(2, 20, 16)


MENAGERIE_ENTRIES = [
    ("NeuralLog", build_neurallog, example_input_neurallog, 2021, MENAGERIE_ZOO),
]
