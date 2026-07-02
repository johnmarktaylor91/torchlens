# SOURCE: vendored from https://github.com/jogecodes/transformerAD @ main
# (utils/transformer.py + utils/attention.py + utils/layers.py)
#
# CAN-Transformer IDS (community PyTorch implementation of a Transformer-based
# intrusion detection system for CAN-bus traffic, following the "Attention is
# All You Need" encoder-decoder recipe applied to windowed CAN-frame sequences
# for next-frame prediction; large prediction error flags anomalous/attack
# traffic -- cf. arXiv:2103.16329 for the CAN-bus transformer-IDS problem
# setting this repo implements). A standard pre-LN-free (post-residual, custom
# `Norm`) Transformer encoder-decoder: sinusoidal positional encoding,
# selectable single-head/general/multi-head scaled-dot-product attention,
# feed-forward sublayers, causal (`nopeak`) masking in the decoder, and a
# final linear layer collapsing the decoder's window output to one predicted
# CAN-frame vector.
#
# Vendored real repo code (Encoder, Decoder, Transformer, PositionalEncoder,
# scaled_DPattention, GeneralAttention, SingleHeadAttention, MultiHeadAttention,
# FeedForward, Norm, EncoderLayer, DecoderLayer, get_clones). Only the
# `.to(torch.device(device))` calls threaded through every submodule
# constructor were left as-is (they are a no-op on the CPU device string used
# here) and the `tqdm`-based `train_model`/`detect` training-loop methods were
# dropped -- none of that is architecture. No layer, attention mechanism, or
# dataflow inside the Transformer itself was changed.

import copy
import math

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils/attention.py
# ---------------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, window, device, encoding_length=10000):
        super().__init__()
        self.d_model = d_model
        self.window = window
        pe = torch.zeros(window, d_model).to(torch.device(device))
        for pos in range(window):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (encoding_length ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (encoding_length ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


def scaled_DPattention(q, k, v, d_attention, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class GeneralAttention(nn.Module):
    def __init__(self, d_model, device, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout).to(torch.device(device))

    def forward(self, q, k, v, mask=None):
        scores = scaled_DPattention(q, k, v, self.d_model, mask, self.dropout)
        output = scores
        return output


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, device, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.out = nn.Linear(d_model, d_model).to(torch.device(device))

    def forward(self, q, k, v, mask=None):
        q_linear = self.q_linear(q)
        k_linear = self.k_linear(k)
        v_linear = self.v_linear(v)
        scores = scaled_DPattention(q_linear, k_linear, v_linear, self.d_model, mask, self.dropout)
        output = self.out(scores)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.N_heads = heads
        self.d_head = (d_model // heads) + 1
        self.q_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.v_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.k_linear = nn.Linear(d_model, self.N_heads * self.d_head)
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.out = nn.Linear(self.N_heads * self.d_head, d_model).to(torch.device(device))

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        q_lin = self.q_linear(q)
        k_lin = self.k_linear(k)
        v_lin = self.v_linear(v)

        q_view = q_lin.view(batch_size, q_len, self.N_heads, self.d_head)
        k_view = k_lin.view(batch_size, k_len, self.N_heads, self.d_head)
        v_view = v_lin.view(batch_size, v_len, self.N_heads, self.d_head)

        q_heads = q_view.transpose(1, 2)
        k_heads = k_view.transpose(1, 2)
        v_heads = v_view.transpose(1, 2)

        scores = scaled_DPattention(q_heads, k_heads, v_heads, self.d_head, mask, self.dropout)

        concat = (
            scores.transpose(1, 2).contiguous().view(batch_size, -1, self.N_heads * self.d_head)
        )

        output = self.out(concat)

        return output


# ---------------------------------------------------------------------------
# utils/layers.py
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, device, dropout, d_ff):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff).to(torch.device(device))
        self.dropout = nn.Dropout(dropout).to(torch.device(device))
        self.linear_2 = nn.Linear(d_ff, d_model).to(torch.device(device))
        self.device = device

    def forward(self, x):
        x = self.dropout(nn.functional.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, device, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size).to(torch.device(device)))
        self.bias = nn.Parameter(torch.zeros(self.size).to(torch.device(device)))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=(0, 1), keepdim=True)
        x_std = x.std(dim=(0, 1), keepdim=True)
        norm = self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, attention, device, dropout, d_ff):
        super().__init__()
        self.norm_1 = Norm(d_model, device).to(torch.device(device))
        self.norm_2 = Norm(d_model, device).to(torch.device(device))

        if attention == "general":
            self.attn = GeneralAttention(d_model, device).to(torch.device(device))
        elif attention == "single":
            self.attn = SingleHeadAttention(d_model, device).to(torch.device(device))
        elif isinstance(attention, int):
            heads = attention
            self.attn = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
        else:
            raise ValueError("Attention type not recognized")

        self.ff = FeedForward(d_model, device, dropout, d_ff).to(torch.device(device))
        self.dropout_1 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_2 = nn.Dropout(dropout).to(torch.device(device))

    def forward(self, x):
        x_norm_1 = self.norm_1(x)
        x_dropout_1 = x + self.dropout_1(self.attn(x_norm_1, x_norm_1, x_norm_1))
        x_norm_2 = self.norm_2(x_dropout_1)
        x_dropout_2 = x_dropout_1 + self.dropout_2(self.ff(x_norm_2))
        return x_dropout_2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, attention, device, dropout, d_ff):
        super().__init__()
        self.norm_1 = Norm(d_model, device).to(torch.device(device))
        self.norm_2 = Norm(d_model, device).to(torch.device(device))
        self.norm_3 = Norm(d_model, device).to(torch.device(device))

        self.dropout_1 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_2 = nn.Dropout(dropout).to(torch.device(device))
        self.dropout_3 = nn.Dropout(dropout).to(torch.device(device))

        if attention == "general":
            self.attn_1 = GeneralAttention(d_model, device).to(torch.device(device))
            self.attn_2 = GeneralAttention(d_model, device).to(torch.device(device))
        elif attention == "single":
            self.attn_1 = SingleHeadAttention(d_model, device).to(torch.device(device))
            self.attn_2 = SingleHeadAttention(d_model, device).to(torch.device(device))
        elif isinstance(attention, int):
            heads = attention
            self.attn_1 = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
            self.attn_2 = MultiHeadAttention(heads, d_model, device).to(torch.device(device))
        else:
            raise ValueError("Attention type not recognized")

        self.ff = FeedForward(d_model, device, dropout, d_ff).to(torch.device(device))

    def forward(self, x, e_outputs, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N_layers)])


# ---------------------------------------------------------------------------
# utils/transformer.py
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, window, device)
        self.layers = get_clones(EncoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)

    def forward(self, src):
        x = self.pe(src)
        for i in range(self.N_layers):
            x = self.layers[i](x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, window, device)
        self.layers = get_clones(DecoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)

    def forward(self, trg, e_outputs, mask):
        x = self.pe(trg)
        for i in range(self.N_layers):
            x = self.layers[i](x, e_outputs, mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout=0.1, d_ff=512):
        super().__init__()
        self.encoder = Encoder(d_model, N_layers, attention, window, device, dropout, d_ff)
        self.decoder = Decoder(d_model, N_layers, attention, window, device, dropout, d_ff)
        self.out = nn.Linear(d_model * (window - 1), d_model).to(torch.device(device))
        self.window = window
        self.device = device

    def nopeak_mask(self, size):
        np_mask = torch.triu(torch.ones(1, size, size), diagonal=1).to(torch.uint8)
        np_mask = (np_mask == 0).to(torch.device(self.device))
        return np_mask

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, self.nopeak_mask(trg.size(1)))
        output = self.out(d_output.view(d_output.size(0), -1))
        return output


def build_can_transformer_ids():
    # Tiny config: d_model=8 (CAN-frame feature vector width), window=5
    # (sliding-window sequence length, so decoder input has window-1=4 steps
    # per the real repo's train_model split), 2 encoder/decoder layers,
    # multi-head attention with 2 heads, small feed-forward width.
    model = Transformer(
        d_model=8, N_layers=2, attention=2, window=5, device="cpu", dropout=0.0, d_ff=16
    )
    model.initialize()
    return model


def example_input_can_transformer_ids():
    # Real repo's forward(src, trg) takes two positional tensor args: src is
    # the full window (batch, window, d_model); trg is the decoder input, the
    # window minus the last (predicted) step (batch, window-1, d_model) --
    # matching `train_model`'s `d_input = batch[:, :-1]` split. Returned as a
    # tuple so torchlens' `input_args` unpacks it as (src, trg) positional
    # args to the real Transformer.forward, unmodified.
    src = torch.randn(3, 5, 8)
    trg = torch.randn(3, 4, 8)
    return (src, trg)


MENAGERIE_ENTRIES = [
    (
        "CAN-Transformer-IDS",
        build_can_transformer_ids,
        example_input_can_transformer_ids,
        2021,
        MENAGERIE_ZOO,
    ),
]
