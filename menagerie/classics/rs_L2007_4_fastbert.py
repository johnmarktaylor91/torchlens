# SOURCE: vendored from autoliuweijie/FastBERT @ 859632f67eb97b1624b26c8f8766972153e6382b
# (pypi/fastbert/fastbert.py::BertMiniClassifier + the real UER BERT stack it wraps around:
# pypi/fastbert/uer/models/model.py::Model, uer/encoders/bert_encoder.py::BertEncoder,
# uer/layers/{embeddings,transformer,multi_headed_attn,position_ffn,layer_norm}.py) --
# the real BertMiniClassifier per-layer "student" early-exit classifier head (the FastBERT
# contribution: Linear -> self-attention pooling -> Linear -> Linear, attached to every
# transformer layer's hidden state for self-distillation early-exit inference) and the
# real UER BertEmbedding/TransformerLayer/BertEncoder stack it sits on top of are preserved
# verbatim. The original `fastbert.py::FastBERT` class itself is NOT vendored: it is a
# heavyweight end-to-end trainer (vocab loading, WordPiece tokenizer, pretrained-checkpoint
# downloading, `.fit()`/`.predict()` training loops via `uer.model_builder.build_model`)
# with no architectural content beyond wrapping this same UER kernel + classifier stack, so
# it is reproduced here as a thin `FastBertClassifier` composition of the real kernel +
# real per-layer classifiers (module wiring only, no new layers/mechanisms). Package-
# relative `uer.*` imports are inlined into this single file since the original multi-file
# `uer/` package layout is not vendored. No architectural change.
"""FastBERT: self-distilling BERT with adaptive early-exit inference.

Liu, Weijie, et al. "FastBERT: a Self-distilling BERT with Adaptive Inference Time."
ACL 2020. Attaches a lightweight "student" classifier (BertMiniClassifier: a lite
self-attention-pooled MLP head) to the hidden state emitted by every transformer layer of
a standard BERT encoder, self-distilled from the final-layer "teacher" classifier so that,
at inference, easy samples can exit early once a shallow-layer student's prediction is
confident enough (uncertainty below a threshold), skipping the remaining layers.
"""

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

__all__ = ["FastBertClassifier"]


# ========== uer/layers/layer_norm.py (verbatim) ==========
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# ========== uer/utils/act_fun.py (verbatim) ==========
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ========== uer/layers/embeddings.py (verbatim, BertEmbedding only) ==========
class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """

    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


# ========== uer/layers/multi_headed_attn.py (verbatim) ==========
class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)

        query, key, value = [
            lin(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
            for lin, x in zip(self.linear_layers, (query, key, value))
        ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)

        return output


# ========== uer/layers/position_ffn.py (verbatim) ==========
class PositionwiseFeedForward(nn.Module):
    """Feed Forward Layer"""

    def __init__(self, hidden_size, feedforward_size):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)

    def forward(self, x):
        inter = gelu(self.linear_1(x))
        output = self.linear_2(inter)
        return output


# ========== uer/layers/transformer.py (verbatim) ==========
class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """

    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, args.dropout)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(args.hidden_size, args.feedforward_size)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)
        return output


# ========== uer/encoders/bert_encoder.py (verbatim) ==========
class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([TransformerLayer(args) for _ in range(self.layers_num)])

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden


# ========== fastbert.py::BertMiniClassifier (verbatim) ==========
class BertMiniClassifier(nn.Module):
    """
    The FastBERT per-layer "student" classifier: a lightweight self-attention-pooled MLP
    head attached to every transformer layer's hidden state, self-distilled from the final
    (teacher) layer so that easy inputs can exit early during inference.
    """

    def __init__(self, args, input_size, labels_num):
        super(BertMiniClassifier, self).__init__()
        self.input_size = input_size
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.labels_num = labels_num
        self.pooling = args.pooling
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        self.self_atten = MultiHeadedAttention(
            self.cla_hidden_size, self.cla_heads_num, args.dropout
        )
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)

    def forward(self, hidden, mask):
        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_atten(hidden, hidden, hidden, mask)

        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]

        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        return logits


class _Args:
    """Plain attribute bag mirroring the subset of FastBERT's `self.args` config object
    consumed by BertEmbedding / TransformerLayer / BertEncoder / BertMiniClassifier."""

    def __init__(
        self, emb_size, hidden_size, feedforward_size, heads_num, layers_num, dropout, pooling
    ):
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.feedforward_size = feedforward_size
        self.heads_num = heads_num
        self.layers_num = layers_num
        self.dropout = dropout
        self.pooling = pooling


class FastBertClassifier(nn.Module):
    """Thin composition of the real UER BERT kernel + one real BertMiniClassifier per
    transformer layer, mirroring `fastbert.py::FastBERT.__init__`'s
    `self.kernel = build_model(...)` + `self.classifiers = nn.ModuleList([...])` wiring
    (module construction only; no new layer types)."""

    def __init__(self, args, vocab_size, labels_num):
        super().__init__()
        self.embedding = BertEmbedding(args, vocab_size)
        self.encoder = BertEncoder(args)
        self.classifiers = nn.ModuleList(
            [BertMiniClassifier(args, args.hidden_size, labels_num) for _ in range(args.layers_num)]
        )

    def forward(self, src, seg):
        seq_length = seg.size(1)
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
        mask = (1.0 - mask) * -10000.0

        hidden = self.embedding(src, seg)
        logits_per_layer = []
        for i in range(self.encoder.layers_num):
            hidden = self.encoder.transformer[i](hidden, mask)
            logits_per_layer.append(self.classifiers[i](hidden, mask))
        return logits_per_layer


def _tiny_args():
    return _Args(
        emb_size=32,
        hidden_size=32,
        feedforward_size=64,
        heads_num=4,
        layers_num=3,
        dropout=0.0,
        pooling="first",
    )


def build_fastbert_classifier():
    model = FastBertClassifier(_tiny_args(), vocab_size=128, labels_num=4)
    model.eval()
    return model


def example_input_fastbert_classifier():
    src = torch.randint(0, 128, (2, 12))
    seg = torch.ones(2, 12, dtype=torch.long)
    return src, seg


MENAGERIE_ENTRIES = [
    (
        "FastBERT (self-distilling BERT, per-layer early-exit classifiers)",
        build_fastbert_classifier,
        example_input_fastbert_classifier,
        2020,
        "vendored",
    ),
]
