# SOURCE: vendored from https://github.com/knu-lcbc/RetroTRAE @ main
# File vendored (near-verbatim): src/transformer.py (Transformer, Encoder, Decoder,
# EncoderLayer, DecoderLayer, MultiheadAttention, FeedFowardLayer, LayerNormalization,
# PositionalEncoder). Upstream reads model hyperparameters (`d_model`, `num_heads`,
# `num_layers`, `d_k`, `d_ff`, `drop_out_rate`, `seq_len`, `device`) as bare module-level
# globals via `from parameters import *`; here they are threaded through the constructors
# instead so the module is self-contained (no behavior change -- same computation graph).
#
# RetroTRAE (Ucak et al., Nat. Commun. 2022) predicts single-step retrosynthesis reactants
# from a product SMILES using the UNMODIFIED vanilla "Attention Is All You Need" Transformer
# encoder-decoder; the paper's actual contribution is a novel fragment-based ("TRAE") SMILES
# tokenization scheme applied to that standard architecture, not an architectural change.
from __future__ import annotations

import math

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        return self.layer(x)


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe_matrix = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)
        self.d_model = d_model
        self.register_buffer("positional_encoding", pe_matrix)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, : x.size(1)]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, drop_out_rate):
        super().__init__()
        self.inf = 1e9
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_values = self.self_attention(q, k, v, mask=mask)
        concat_output = (
            attn_values.transpose(1, 2).contiguous().view(input_shape[0], -1, self.d_model)
        )

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1 * self.inf)

        attn_distribs = self.attn_softmax(attn_scores)
        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self, d_model, d_ff, drop_out_rate):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_ff, drop_out_rate):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, d_k, drop_out_rate)
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, drop_out_rate)
        self.drop_out_2 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x)
        x = x + self.drop_out_1(self.multihead_attention(x_1, x_1, x_1, mask=e_mask))
        x_2 = self.layer_norm_2(x)
        x = x + self.drop_out_2(self.feed_forward(x_2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_ff, drop_out_rate):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model)
        self.masked_multihead_attention = MultiheadAttention(d_model, num_heads, d_k, drop_out_rate)
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization(d_model)
        self.multihead_attention = MultiheadAttention(d_model, num_heads, d_k, drop_out_rate)
        self.drop_out_2 = nn.Dropout(drop_out_rate)

        self.layer_norm_3 = LayerNormalization(d_model)
        self.feed_forward = FeedFowardLayer(d_model, d_ff, drop_out_rate)
        self.drop_out_3 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_output, e_mask, d_mask):
        x_1 = self.layer_norm_1(x)
        x = x + self.drop_out_1(self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask))
        x_2 = self.layer_norm_2(x)
        x = x + self.drop_out_2(self.multihead_attention(x_2, e_output, e_output, mask=e_mask))
        x_3 = self.layer_norm_3(x)
        x = x + self.drop_out_3(self.feed_forward(x_3))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_k, d_ff, drop_out_rate):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_k, d_ff, drop_out_rate) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)
        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_k, d_ff, drop_out_rate):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_k, d_ff, drop_out_rate) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)
        return self.layer_norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        seq_len=100,
        num_heads=8,
        num_layers=6,
        d_model=512,
        d_ff=2048,
        drop_out_rate=0.1,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        d_k = d_model // num_heads

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(seq_len, d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_k, d_ff, drop_out_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_k, d_ff, drop_out_rate)
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input)
        trg_input = self.positional_encoder(trg_input)

        e_output = self.encoder(src_input, e_mask)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask)

        output = self.softmax(self.output_linear(d_output))

        return output


# ---------------------------------------------------------------------------
# Staging harness (tiny random-init construction + example input)
# ---------------------------------------------------------------------------
def build_retrotrae():
    return Transformer(
        src_vocab_size=64,
        trg_vocab_size=64,
        seq_len=16,
        num_heads=4,
        num_layers=2,
        d_model=16,
        d_ff=32,
        drop_out_rate=0.0,
    ).eval()


def example_input_retrotrae():
    batch, seq_len, vocab_size = 2, 8, 64
    src_input = torch.randint(1, vocab_size, (batch, seq_len), dtype=torch.long)
    trg_input = torch.randint(1, vocab_size, (batch, seq_len), dtype=torch.long)
    # encoder self-attn padding mask: (B, 1, L), all-valid (no padding) for this tiny example
    e_mask = torch.ones(batch, 1, seq_len, dtype=torch.bool)
    # decoder self-attn causal + padding mask: (B, L, L)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    d_mask = causal.unsqueeze(0).expand(batch, -1, -1)
    return (src_input, trg_input, e_mask, d_mask)


MENAGERIE_ENTRIES = [
    ("RetroTRAE", build_retrotrae, example_input_retrotrae, 2022, MENAGERIE_ZOO),
]
