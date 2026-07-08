# SOURCE: vendored from https://github.com/HelenGuohx/logbert @ master
# (bert_pytorch/model/{bert.py, language_model.py, log_model.py, transformer.py,
#  embedding/*, attention/*, utils/*}). LogBERT: BERT-based masked log-key
# language model for log anomaly detection (Guo, Yuan, Wu, Huang, Wang, IJCNN
# 2021, "LogBERT: Log Anomaly Detection via BERT").
#
# Vendored real repo code: the BERT encoder stack (BERT/TransformerBlock/
# MultiHeadedAttention/Attention/PositionwiseFeedForward/GELU/LayerNorm/
# SublayerConnection) and the embedding stack (BERTEmbedding = Token +
# Positional + Segment + Time embeddings) are copied verbatim from the real
# repo files (only merged into one module and import paths flattened -- no
# layer, dimension, or dataflow was changed). BERTLog is the real inference
# head used by predict_log.py (masked-log-key prediction + a raw CLS pooling
# output), kept as-is; BERTLM (the pretraining head with next-sentence +
# masked-LM) is also vendored since it is the real pretraining architecture
# used by trainer/pretrain.py.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# bert_pytorch/model/utils/*
# ---------------------------------------------------------------------------
class GELU(nn.Module):
    """Paper Section 3.4, last paragraph notice that BERT used the GELU
    instead of RELU."""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


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


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


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


# ---------------------------------------------------------------------------
# bert_pytorch/model/attention/*
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    "Compute 'Scaled Dot Product Attention"

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."

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


# ---------------------------------------------------------------------------
# bert_pytorch/model/transformer.py
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection."""

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
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


# ---------------------------------------------------------------------------
# bert_pytorch/model/embedding/*
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_size)

    def forward(self, time_interval):
        return self.time_embed(time_interval)


class BERTEmbedding(nn.Module):
    """BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        3. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
    sum of all these features are output of BERTEmbedding."""

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, is_logkey=True, is_time=False):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time

    def forward(self, sequence, segment_label=None, time_info=None):
        x = self.position(sequence)
        # if self.is_logkey:
        x = x + self.token(sequence)
        if segment_label is not None:
            x = x + self.segment(segment_label)
        if self.is_time:
            x = x + self.time_embed(time_info)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# bert_pytorch/model/bert.py
# ---------------------------------------------------------------------------
class BERT(nn.Module):
    """BERT model : Bidirectional Encoder Representations from Transformers."""

    def __init__(
        self,
        vocab_size,
        max_len=512,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        is_logkey=True,
        is_time=False,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden,
            max_len=max_len,
            is_logkey=is_logkey,
            is_time=is_time,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 2, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info=None, time_info=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info, time_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


# ---------------------------------------------------------------------------
# bert_pytorch/model/log_model.py -- real inference head (predict_log.py)
# ---------------------------------------------------------------------------
class MaskedLogModel(nn.Module):
    """predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size."""

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class TimeLogModel(nn.Module):
    def __init__(self, hidden, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidden, time_size)

    def forward(self, x):
        return self.linear(x)


class BERTLog(nn.Module):
    """BERT Log Model."""

    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        self.time_lm = TimeLogModel(self.bert.hidden)

    def forward(self, x, time_info):
        x = self.bert(x, time_info=time_info)
        logkey_output = self.mask_lm(x)
        cls_output = x[:, 0]
        return {"logkey_output": logkey_output, "cls_output": cls_output}


def build_logbert():
    vocab_size = 32
    bert = BERT(
        vocab_size=vocab_size, max_len=16, hidden=64, n_layers=2, attn_heads=4, is_time=True
    )
    return BERTLog(bert, vocab_size=vocab_size)


def example_input_logbert():
    # (x, time_info) positional args to BERTLog.forward; x is a batch of
    # integer log-key-id sequences, time_info the per-step time-interval
    # feature consumed by the (is_time=True) BERTEmbedding path.
    x = torch.randint(1, 32, (2, 16))
    time_info = torch.rand(2, 16, 1)
    return (x, time_info)


MENAGERIE_ENTRIES = [
    (
        "LogBERT",
        build_logbert,
        example_input_logbert,
        2021,
        MENAGERIE_ZOO,
    ),
]
