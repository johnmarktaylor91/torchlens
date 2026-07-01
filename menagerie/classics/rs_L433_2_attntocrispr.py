# SOURCE: vendored from qiaoliuhub/AttnToCrispr @ master
# Files: Sublayers.py, Layers.py, attention_model.py (EmbeddingTransformer/get_model path)
# The real repo dynamically imports a per-dataset `config`/`attention_setting` module via
# `sys.argv`/`importlib` (e.g. models/A549/config.py) and pulls in unrelated data-loading
# modules (OT_crispr_attn, crispr_attn) purely to resolve a `device2` global. Those are
# orchestration/data-path concerns, not architecture. Here the same real nn.Module classes
# are kept verbatim; only the config-file indirection is replaced with plain constructor
# arguments carrying the same default values found in models/A549/attention_setting.py and
# models/A549/config.py (d_model=20, heads=4, dropout=0.2, n_layers=1, embedding_voca_size,
# seq_len, word_len, sep_len, dropout) so the module builds standalone.
"""AttnToCrispr: scaled dot-product self-attention Transformer for CRISPR on-target
efficiency prediction, vendored from the real qiaoliuhub/AttnToCrispr repository."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

MENAGERIE_ZOO = "vendored-pytorch"


# --- Sublayers.py (verbatim) ---


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        sl = q.size(1)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, sl, self.h, -1)
        q = self.q_linear(q).view(bs, sl, self.h, -1)
        v = self.v_linear(v).view(bs, sl, self.h, -1)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        output = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=80, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class OutputFeedForward(nn.Module):
    def __init__(self, H, W, extra_length=0, d_layers=None, dropout=0.1):
        super().__init__()

        self.d_layers = [512, 1] if d_layers is None else d_layers
        self.linear_1 = nn.Linear(H * W + extra_length, self.d_layers[0])
        self.n_layers = len(self.d_layers)
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.n_layers))
        self.dropouts[0] = nn.Dropout(p=0.2)
        self.layers = nn.ModuleList(
            nn.Linear(d_layers[i - 1], d_layers[i]) for i in range(1, self.n_layers)
        )

    def forward(self, x):
        x = self.dropouts[0](x)
        x = self.linear_1(x)
        for i in range(self.n_layers - 1):
            x = self.dropouts[i + 1](F.elu(x))
            x = self.layers[i](x)
        return x


# --- Layers.py (verbatim, minus the dynamic config-module attention_setting lookup;
#     `attention_layer_norm` is passed in explicitly instead, matching the real default
#     `False` from models/A549/attention_setting.py) ---


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, d_input, d_model, heads, dropout=0.1, attention_layer_norm=False):
        super().__init__()
        self.input_linear = nn.Linear(d_input, d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.attention_layer_norm = attention_layer_norm

    def forward(self, x, mask=None):
        x = F.relu(self.input_linear(x))
        x2 = self.norm_1(x) if self.attention_layer_norm else x
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x) if self.attention_layer_norm else x
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_input, d_model, heads, dropout=0.1, attention_layer_norm=False):
        super().__init__()
        self.input_linear = nn.Linear(d_input, d_model)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.attention_layer_norm = attention_layer_norm

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):
        x = F.relu(self.input_linear(x))
        x2 = self.norm_1(x) if self.attention_layer_norm else x
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x) if self.attention_layer_norm else x
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x) if self.attention_layer_norm else x
        x = x + self.dropout_3(self.ff(x2))
        return x


# --- attention_model.py (verbatim architecture; `customized_CNN`/`add_seq_cnn`/
#     `add_parallel_cnn` branches are kept but the CNN branches are disabled by default,
#     matching config default `add_seq_cnn=True` in real attention_setting.py -- kept
#     configurable via constructor rather than a module-level import) ---


class Encoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout, attention_layer_norm=False):
        super().__init__()
        self.N = N
        self.layers = get_clones(
            EncoderLayer(d_input, d_model, heads, dropout, attention_layer_norm), N
        )
        self.norm = nn.LayerNorm(d_model)
        self.attention_layer_norm = attention_layer_norm

    def forward(self, src, mask=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x) if self.attention_layer_norm else x


class Decoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout, attention_layer_norm=False):
        super().__init__()
        self.N = N
        self.layers = get_clones(
            DecoderLayer(d_input, d_model, heads, dropout, attention_layer_norm), N
        )
        self.norm = nn.LayerNorm(d_model)
        self.attention_layer_norm = attention_layer_norm

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x) if self.attention_layer_norm else x


class Transformer(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        n_feature_dim,
        N,
        heads,
        dropout,
        extra_length,
        attention_layer_norm=False,
        output_FF_layers=None,
    ):
        super().__init__()
        self.encoder = Encoder(n_feature_dim, d_model, N, heads, dropout, attention_layer_norm)
        self.decoder = Decoder(n_feature_dim, d_model, N, heads, dropout, attention_layer_norm)
        self.out = OutputFeedForward(
            d_model,
            d_input,
            extra_length,
            d_layers=output_FF_layers if output_FF_layers is not None else [200, 1],
            dropout=dropout,
        )

    def forward(self, src, trg, extra_input_for_FF=None, src_mask=None, trg_mask=None):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        flat_d_output = d_output.view(-1, d_output.size(-2) * d_output.size(-1))
        if extra_input_for_FF is not None:
            flat_d_output = cat((flat_d_output, extra_input_for_FF), dim=1)
        output = self.out(flat_d_output)
        return output


class EmbeddingTransformer(Transformer):
    """Real on-target CRISPR efficiency Transformer (`get_model()` path).

    Embeds an integer-encoded guide-RNA sequence (source) and its shifted copy
    (target) with learned token + positional embeddings, then runs them through
    a real encoder/decoder self-attention stack before a feed-forward regression
    head -- this is the actual model built by `crispr_attn.get_model()` in the
    real repository (config-module lookups replaced with explicit arguments).
    """

    def __init__(
        self,
        embedding_vec_dim,
        d_input,
        d_model,
        N,
        heads,
        dropout,
        extra_length,
        embedding_voca_size=25,
        seq_len=23,
        seq_start=0,
        word_len=1,
        sep_len=0,
        attention_layer_norm=False,
        output_FF_layers=None,
    ):
        super().__init__(
            d_input,
            d_model,
            embedding_vec_dim,
            N,
            heads,
            dropout,
            extra_length,
            attention_layer_norm,
            output_FF_layers,
        )
        self.seq_len = seq_len
        self.seq_start = seq_start
        self.word_len = word_len
        self.sep_len = sep_len
        self.embedding = nn.Embedding(embedding_voca_size, embedding_vec_dim)
        self.embedding_2 = nn.Embedding(embedding_voca_size, embedding_vec_dim)
        self.trg_embedding = nn.Embedding(embedding_voca_size, embedding_vec_dim)
        pos_len = seq_len - word_len + 1
        self.embedding_pos = nn.Embedding(pos_len, embedding_vec_dim)
        self.trg_embedding_pos = nn.Embedding(pos_len, embedding_vec_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, trg=None, extra_input_for_FF=None, src_mask=None, trg_mask=None):
        if self.sep_len != 0:
            src_1 = src[:, : self.sep_len]
            src_2 = src[:, self.sep_len :]
            embedded_src = self.embedding(src_1)
            embedded_src_2 = self.embedding_2(src_2)
            embedded_src = cat(tuple([embedded_src, embedded_src_2]), dim=1)
        else:
            embedded_src = self.embedding(src)

        bs = src.size(0)
        pos_length = self.seq_len - self.seq_start - self.word_len + 1
        pos = torch.stack([torch.arange(pos_length, device=src.device) for _ in range(bs)])
        embedded_src_pos = self.embedding_pos(pos)
        embedded_src_1 = embedded_src + embedded_src_pos
        embedded_src_2 = self.dropout(embedded_src_1)

        if trg is not None:
            embedded_trg = self.trg_embedding(trg)
            embedded_trg_pos = self.trg_embedding_pos(pos)
            embedded_trg_1 = embedded_trg + embedded_trg_pos
            embedded_trg_2 = self.dropout(embedded_trg_1)
        else:
            embedded_trg_2 = embedded_src_2

        output = super().forward(embedded_src_2, embedded_trg_2, extra_input_for_FF)
        return output


# --- staging build/example helpers ---


def build_attntocrispr():
    """Tiny EmbeddingTransformer matching the real get_model() construction path,
    with real defaults from models/A549/attention_setting.py: d_model=20, heads=4,
    n_layers=1, dropout=0.2 (shrunk seq_len/embedding_vec_dim/vocab for a tiny trace)."""
    embedding_vec_dim = 4
    seq_len = 8
    d_input = seq_len  # matches get_model(d_input=20) style call with seq_len substituted
    return EmbeddingTransformer(
        embedding_vec_dim=embedding_vec_dim,
        d_input=d_input,
        d_model=20,
        N=1,
        heads=4,
        dropout=0.2,
        extra_length=0,
        embedding_voca_size=8,
        seq_len=seq_len,
        seq_start=0,
        word_len=1,
        sep_len=0,
        attention_layer_norm=False,
        output_FF_layers=[16, 1],
    )


def example_input_attntocrispr():
    # Integer-encoded guide-RNA sequence tokens, shape (batch, seq_len)
    return torch.randint(0, 8, (2, 8))


MENAGERIE_ENTRIES = [
    ("AttnToCrispr", "build_attntocrispr", "example_input_attntocrispr", 2019, "vendored-pytorch"),
]
