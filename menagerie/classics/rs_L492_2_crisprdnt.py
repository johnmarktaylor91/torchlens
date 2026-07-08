# FAITHFUL PORT of gzrgzx/CrisprDNT @ main (original framework: TensorFlow 2.3 / Keras,
# code/model/model_network.py::new_crispr_ip + PositionalEncoding)
#
# CrisprDNT ("Transformer-based Anti-noise Model for Predicting CRISPR-Cas9 Off-Target
# Activities") is real published TF/Keras code, but it depends on TF1-flavored Keras
# internals (`tensorflow.python.keras.*`) plus non-base third-party packages
# (`keras_multi_head.MultiHeadAttention`, `keras_bert`, `keras_layer_normalization`) that
# are not installed and not reasonably installable alongside our torch-only base env. This
# module transcribes the `new_crispr_ip` architecture (the CrisprDNT backbone) faithfully
# into self-contained torch: 14x23 mismatch-type-encoded guide/target pair -> per-position
# Conv2D branch (channels_first, kernel spans the full 14-channel axis) -> BatchNorm ->
# average+max 1D pooling of the conv features -> concat with the raw per-position encoding
# -> BiLSTM -> LayerNorm -> sinusoidal positional encoding -> two pre-norm-residual
# transformer blocks (multi-head self-attention + 2-layer FFN, each with a residual-then-
# LayerNorm exactly as in the original, i.e. post-norm over the sum) -> flatten -> dense
# classifier head (256 -> 64 -> 2, softmax in the original; kept as raw logits here since
# TorchLens traces functional graphs, not fit loops).
import math

import torch
import torch.nn.functional as f
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

SEQ_LEN = 23
N_CHANNELS = 14
CONV_FILTERS = 64
LSTM_HIDDEN = 32
FFN_HIDDEN = 512
N_HEADS = 8


class SinusoidalPositionalEncoding(nn.Module):
    """Port of CrisprDNT's `PositionalEncoding` Keras layer: additive sin/cos table."""

    def __init__(self, sequence_len: int, embedding_dim: int):
        super().__init__()
        position = torch.arange(sequence_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(
            10000.0,
            (2.0 * torch.arange(embedding_dim, dtype=torch.float32) / embedding_dim),
        )
        table = position / div_term
        table[:, 0::2] = torch.sin(table[:, 0::2])
        table[:, 1::2] = torch.cos(table[:, 1::2])
        self.register_buffer("table", table.unsqueeze(0))

    def forward(self, x):
        return x + self.table


class MultiHeadSelfAttention(nn.Module):
    """Port of `keras_multi_head.MultiHeadAttention(head_num=8)` used with q=k=v."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = f.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Port of the residual attention+FFN stanza repeated twice in `new_crispr_ip`."""

    def __init__(self, embed_dim: int, ffn_hidden: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, ffn_hidden)
        self.ffn2 = nn.Linear(ffn_hidden, embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attn(x)
        residual1 = attn_out + x
        norm1 = self.norm_attn(residual1)

        ffn_out = self.ffn2(f.relu(self.ffn1(norm1)))
        residual2 = norm1 + ffn_out
        norm2 = self.norm_ffn(residual2)
        return norm2


class CrisprDNT(nn.Module):
    """Faithful port of `new_crispr_ip` from CrisprDNT/code/model/model_network.py."""

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        n_channels: int = N_CHANNELS,
        conv_filters: int = CONV_FILTERS,
        lstm_hidden: int = LSTM_HIDDEN,
        ffn_hidden: int = FFN_HIDDEN,
        n_heads: int = N_HEADS,
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels

        # conv_1_output = Conv2D(64, (1, n_channels), data_format='channels_first')(input_value)
        # equivalent in torch (batch, 1, n_channels, seq_len) -> (batch, conv_filters, 1, seq_len)
        self.conv = nn.Conv2d(1, conv_filters, kernel_size=(n_channels, 1))
        self.conv_bn = nn.BatchNorm2d(conv_filters)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        bilstm_input_dim = n_channels + 2 * conv_filters
        self.bilstm = nn.LSTM(
            input_size=bilstm_input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        embed_dim = 2 * lstm_hidden
        self.bilstm_norm = nn.LayerNorm(embed_dim)

        self.pos_encoding = SinusoidalPositionalEncoding(seq_len, embed_dim)

        self.block1 = TransformerBlock(embed_dim, ffn_hidden, n_heads)
        self.block2 = TransformerBlock(embed_dim, ffn_hidden, n_heads)

        flat_dim = seq_len * embed_dim
        self.head1 = nn.Linear(flat_dim, 256)
        self.head2 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.25)
        self.head3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, n_channels, seq_len) -- mismatch-type-encoded guide/target pair
        batch = x.shape[0]
        conv_in = x.unsqueeze(1)  # (batch, 1, n_channels, seq_len)
        conv_out = f.relu(self.conv(conv_in))  # (batch, conv_filters, 1, seq_len)
        conv_out = self.conv_bn(conv_out)
        conv_out = conv_out.squeeze(2)  # (batch, conv_filters, seq_len)

        conv_avg = self.avg_pool(conv_out)
        conv_avg = f.interpolate(conv_avg, size=self.seq_len, mode="nearest")
        conv_max = self.max_pool(conv_out)
        conv_max = f.interpolate(conv_max, size=self.seq_len, mode="nearest")

        raw_seq = x.transpose(1, 2)  # (batch, seq_len, n_channels)
        conv_avg_seq = conv_avg.transpose(1, 2)  # (batch, seq_len, conv_filters)
        conv_max_seq = conv_max.transpose(1, 2)

        merged = torch.cat([raw_seq, conv_avg_seq, conv_max_seq], dim=-1)
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.bilstm_norm(lstm_out)

        pos_embedded = self.pos_encoding(lstm_out)
        block1_out = self.block1(pos_embedded)
        block2_out = self.block2(block1_out)

        flat = block2_out.reshape(batch, -1)
        hidden1 = f.relu(self.head1(flat))
        hidden2 = f.relu(self.head2(hidden1))
        hidden2 = self.dropout(hidden2)
        logits = self.head3(hidden2)
        return logits


def build_crisprdnt():
    return CrisprDNT()


def example_input_crisprdnt():
    return torch.randn(2, N_CHANNELS, SEQ_LEN)


MENAGERIE_ENTRIES = [
    ("CrisprDNT", build_crisprdnt, example_input_crisprdnt, 2023, "REIMPLEMENT"),
]
