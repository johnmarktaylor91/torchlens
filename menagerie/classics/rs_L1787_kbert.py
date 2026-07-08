# SOURCE: vendored from autoliuweijie/K-BERT @ master
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/layers/layer_norm.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/layers/embeddings.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/layers/position_ffn.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/layers/multi_headed_attn.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/layers/transformer.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/encoders/bert_encoder.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/uer/utils/act_fun.py
# https://raw.githubusercontent.com/autoliuweijie/K-BERT/master/run_kbert_cls.py (BertClassifier)
#
# Liu, Zhou, Zhao, Wang, Ju, Deng, Wang 2020 (AAAI) "K-BERT: Enabling Language
# Representation with Knowledge Graph" -- a BERT variant that injects knowledge-graph
# triples directly into the input sentence as extra branch tokens, then uses two
# architectural devices (not just data preprocessing) to keep the KG-injected sentence
# from corrupting normal language modeling: (1) "soft-position" embeddings (`pos`) that
# re-index injected KG-branch tokens to sit at the same position as their anchor token's
# original neighbors, and (2) a "visible matrix" (`vm`) -- a per-example dense mask that
# is *added directly into every transformer layer's attention scores* (see
# `BertEncoder.forward`: `mask = vm.unsqueeze(1); ...; hidden = self.transformer[i](hidden,
# mask)`), restricting each token to only attend to tokens that are visible to it in the
# KG-augmented sentence tree. This visible-matrix-as-attention-mask mechanism is a real
# architectural modification of BERT's self-attention (not merely input formatting), so it
# is vendored rather than represented as a bare `transformers.BertModel` recipe.
#
# `BertEmbedding`, `LayerNorm`, `PositionwiseFeedForward`, `MultiHeadedAttention`,
# `TransformerLayer`, `BertEncoder`, `gelu` are copied verbatim from the real `uer/`
# framework files. `BertClassifier` is copied verbatim from `run_kbert_cls.py` (the
# official K-BERT classification example), which composes the embedding + encoder into
# the full K-BERT-for-classification model used in the paper's downstream tasks. No
# architectural changes were made; only unused training-loop code (loss/CLS-pooling
# branches other than the default) was left untouched (still real code, still runs).
#
# The real model consumes 4 aligned tensors per the K-BERT calling convention in
# `add_knowledge_worker`/`BertClassifier.forward`: `src` (token ids), `mask` (attention
# validity mask, 1/0), `pos` (soft-position ids from the KG injector), and `vm` (the dense
# visible matrix, [batch, seq_len, seq_len], 1 where token i can see token j). We
# reproduce this real multi-tensor calling convention with a tiny synthetic KG-consistent
# visible matrix (block-diagonal-plus-diagonal, mirroring the real injector's guarantee
# that every token can always see itself and its own sentence) instead of running the full
# `brain.KnowledgeGraph` triple-injection pipeline (which needs an external `.spo`
# knowledge-graph file not needed to exercise the real forward pass).

import math

import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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

    def forward(self, src, seg, pos=None):
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(
                torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(word_emb.size(0), 1)
            )
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


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
            linear(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)

        return output


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


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features. K-BERT's
    encoder additionally accepts an explicit "visible matrix" `vm` in place of the plain
    segment mask -- this is the KG-injection mechanism made architectural.
    """

    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([TransformerLayer(args) for _ in range(self.layers_num)])

    def forward(self, emb, seg, vm=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            vm: [batch_size x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if vm is None:
            mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = vm.unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden


class _KBertArgs:
    """Tiny stand-in for the argparse Namespace `uer` passes as `args` everywhere."""

    def __init__(
        self,
        emb_size,
        hidden_size,
        heads_num,
        layers_num,
        feedforward_size,
        dropout,
        labels_num,
        pooling,
    ):
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.layers_num = layers_num
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.labels_num = labels_num
        self.pooling = pooling


class BertClassifier(nn.Module):
    """Copied verbatim (architecture-wise) from run_kbert_cls.py's BertClassifier."""

    def __init__(self, args, embedding, encoder):
        super(BertClassifier, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.use_vm = True

    def forward(self, src, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            mask: [batch_size x seq_length] (segment indicator, matches uer's `seg` arg)
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return logits


def _build_visible_matrix(batch_size, seq_length):
    """Mirrors the real KG injector's guarantee (see brain/knowgraph.py
    `add_knowledge_with_vm`): every token can always see itself and every token in its
    own original sentence span; KG-branch tokens additionally see only their anchor's
    span. We approximate this with a block-diagonal-plus-diagonal visible matrix over two
    synthetic "branches" per example -- structurally the same shape/semantics the real
    vm has (dense 0/1 [batch, seq_len, seq_len] visibility), without running the real
    HowNet .spo triple lookup."""
    vm = torch.eye(seq_length).unsqueeze(0).repeat(batch_size, 1, 1)
    half = seq_length // 2
    vm[:, :half, :half] = 1.0
    vm[:, half:, half:] = 1.0
    return vm


def build_kbert():
    args = _KBertArgs(
        emb_size=32,
        hidden_size=32,
        heads_num=2,
        layers_num=2,
        feedforward_size=64,
        dropout=0.0,
        labels_num=2,
        pooling="first",
    )
    vocab_size = 512
    embedding = BertEmbedding(args, vocab_size)
    encoder = BertEncoder(args)
    return BertClassifier(args, embedding, encoder)


def example_input_kbert():
    batch_size, seq_length = 2, 16
    src = torch.randint(0, 512, (batch_size, seq_length), dtype=torch.long)
    seg = torch.ones(batch_size, seq_length, dtype=torch.long)
    pos = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    vm = _build_visible_matrix(batch_size, seq_length)
    return (src, seg, pos, vm)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("K-BERT for legal judgment", "build_kbert", "example_input_kbert", 2020, "vendored"),
]
