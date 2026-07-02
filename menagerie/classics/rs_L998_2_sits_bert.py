# SOURCE: vendored from linlei1214/SITS-BERT @ master
# (code/model/bert.py + code/model/classification_model.py + code/model/transformer.py +
#  code/model/embedding/bert.py + code/model/embedding/position.py +
#  code/model/attention/multi_head.py + code/model/attention/single.py +
#  code/model/utils/feed_forward.py + code/model/utils/gelu.py +
#  code/model/utils/layer_norm.py + code/model/utils/sublayer.py)
#
# SITS-BERT: a BERT-style Transformer encoder for Satellite Image Time Series (SITS)
# classification. Per-timestep spectral-band observations are linearly embedded and summed
# with a day-of-year sinusoidal positional encoding (BERTEmbedding), then passed through a
# stack of pre-norm Transformer blocks (TransformerBlock: multi-head scaled-dot-product
# self-attention + GELU-activated position-wise FFN, each wrapped in a residual
# SublayerConnection with a custom LayerNorm), and finally max-pooled over the time axis and
# projected to class logits (SBERTClassification / MulticlassClassification). All code below
# is copied verbatim from the real repo (only base-lib deps: torch), collapsed into a single
# file (only import paths were adjusted; relative-package `.` imports replaced with direct
# references since everything lives in one module here).
# Yuan & Lin, "Self-Supervised Pretraining of Transformers for Satellite Image Time Series
# Classification", IEEE JSTARS 2021 (SITS-BERT).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- code/model/utils/gelu.py ---


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# --- code/model/utils/layer_norm.py ---


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# --- code/model/utils/sublayer.py ---


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# --- code/model/utils/feed_forward.py ---


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# --- code/model/attention/single.py ---


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


# --- code/model/attention/multi_head.py ---


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# --- code/model/transformer.py ---


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


# --- code/model/embedding/position.py ---


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()  # [d_model/2,]

        pe[1:, 0::2] = torch.sin(position * div_term)  # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)  # broadcasting to [max_len, d_model/2]

        self.register_buffer("pe", pe)

    def forward(self, doy):
        return self.pe[doy, :]


# --- code/model/embedding/bert.py ---


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a fully connected layer
        2. PositionalEncoding : adding positional information using sin, cos

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, embedding_dim, dropout=0.1):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.input = nn.Linear(in_features=num_features, out_features=embedding_dim)
        self.position = PositionalEncoding(d_model=embedding_dim, max_len=366)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence, doy_sequence):
        batch_size = input_sequence.size(0)
        obs_embed = self.input(input_sequence)  # [batch_size, seq_length, embedding_dim]
        x = obs_embed.repeat(1, 1, 2)  # [batch_size, seq_length, embedding_dim*2]
        for i in range(batch_size):
            x[i, :, self.embed_size :] = self.position(
                doy_sequence[i, :]
            )  # [seq_length, embedding_dim]

        return self.dropout(x)


# --- code/model/bert.py ---


class SBERT(nn.Module):
    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-BERT model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(num_features, int(hidden / 2))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, doy, mask):
        mask = (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(input_sequence=x, doy_sequence=doy)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


# --- code/model/classification_model.py ---


class MulticlassClassification(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.MaxPool1d(64)
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        x = self.linear(x)
        return x


class SBERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes):
        super().__init__()
        self.sbert = sbert
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.classification(x, mask)


def build_sits_bert():
    # tiny config; hidden must be even (embedding splits it into two halves) and divisible
    # by attn_heads for MultiHeadedAttention's d_model % h == 0 assertion.
    sbert = SBERT(num_features=10, hidden=32, n_layers=2, attn_heads=4, dropout=0.1)
    model = SBERTClassification(sbert, num_classes=9)
    model.eval()
    return model


def example_input_sits_bert():
    # x: per-timestep spectral bands [batch, seq_len, num_features]
    # doy: day-of-year index per timestep [batch, seq_len] (used to index the positional table)
    # mask: valid-timestep mask [batch, seq_len] (nonzero = valid, per the `mask > 0` check)
    # seq_len is fixed at 64: MulticlassClassification's real head hardcodes
    # `nn.MaxPool1d(64)`, so the transformer's output sequence length must be exactly 64.
    batch, seq_len, num_features = 2, 64, 10
    x = torch.randn(batch, seq_len, num_features)
    doy = torch.randint(1, 300, (batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    return (x, doy, mask)


MENAGERIE_ENTRIES = [
    ("SITS-BERT", build_sits_bert, example_input_sits_bert, 2021, "vendored-pytorch"),
]
