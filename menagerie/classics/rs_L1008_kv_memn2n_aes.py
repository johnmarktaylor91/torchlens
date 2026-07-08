# FAITHFUL PORT of siyuanzhao/automated-essay-grading @ master (original framework: TensorFlow 1.x)
#
# The real repo (https://github.com/siyuanzhao/automated-essay-grading, paper "A
# Memory-Augmented Neural Model for Automated Grading", L@S 2017) is written against
# TensorFlow 1.10 with `tf.contrib.layers`, `tf.contrib.rnn.static_rnn`, Python-2-only
# `print` statements, and `sklearn.cross_validation` (removed from sklearn since
# 0.20) -- none of which run under the installed TF2/sklearn/torch stack, and the
# TF1.x graph-mode API (`tf.placeholder`, `tf.get_variable`, `tf.contrib.*`) has no
# reasonable install path alongside our torch env. Per the ladder this routes to a
# faithful port: `memn2n_kv.py`'s `MemN2N_KV` class (bow reader; the default and
# the variant train.py actually exercises -- gru is a `reader='gru'` alternative
# path in the same class, not used by default) is transcribed op-for-op into torch.
#
# Architecture (verbatim to the real `MemN2N_KV.__init__` / `_key_addressing` with
# `reader='bow'`):
#   - Frozen GloVe word embeddings (here: a random-init embedding standing in for
#     the frozen `self.W = tf.Variable(self.w_placeholder, trainable=False)` matrix
#     -- the real repo loads actual GloVe vectors at train time; the *lookup +
#     bag-of-words position-encoding pooling* is the architecture, not the specific
#     pretrained values).
#   - `position_encoding(sentence_size, embedding_size)`: the fixed (non-trainable)
#     sinusoid-free bilinear position-encoding matrix from End-To-End Memory
#     Networks (Sukhbaatar et al. 2015, eq. in sec 4.1), applied to the embedded
#     query (essay) and memory keys (a fixed set of exemplar essays, one per score
#     bucket) before summing over the token dimension -- the "bow" (bag-of-words)
#     reader.
#   - Three learned projection matrices A (query -> feature space), A_mkey /
#     A_mvalue (memory key/value -> feature space) and a single shared R matrix
#     reused across all `hops` (the real code builds `r_list` as `hops` references
#     to the SAME `R` variable, not `hops` independent matrices -- reproduced
#     verbatim, including the commented-out per-hop `R_{}` alternative left dead in
#     the original).
#   - `_key_addressing`: iterative "key addressing" over `hops` iterations --
#     projected memory keys dotted against the current query state give softmax
#     attention weights over the memory_key_size exemplar slots ("mem_attention_probs",
#     kept as an output here exactly as in the real repo, since it is the paper's
#     named interpretability signal); attention-weighted memory values are summed and
#     added to the running query state, then passed through R and ReLU to produce the
#     next hop's query state (`u_k = relu(R @ (u[-1] + o_k))`).
#   - Final projection matrix B maps the last hop's feature-space state to
#     `score_range` logits (`o @ B + logits_bias`); softmax gives the predicted
#     score-bucket distribution. The real code's `l2_lambda`-weighted L2 loss and
#     `add_gradient_noise` are training-time-only (loss/optimizer scaffolding, not
#     part of the forward architecture) and are not part of the traced module.
#
# Only base-lib deps: torch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MENAGERIE_ZOO = "ported-pytorch"


def position_encoding(sentence_size: int, embedding_size: int) -> torch.Tensor:
    """Position Encoding described in section 4.1 of Sukhbaatar et al. 2015,
    transcribed verbatim from the real repo's numpy implementation."""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return torch.from_numpy(np.transpose(encoding)).float()


class MemN2N_KV(nn.Module):
    """Key Value Memory Network with bag-of-words (bow) reader, faithfully ported
    from the real repo's TF1.x `MemN2N_KV` class (reader='bow' branch)."""

    def __init__(
        self,
        vocab_size: int,
        query_size: int,
        story_size: int,
        memory_key_size: int,
        memory_value_size: int,
        embedding_size: int,
        score_range: int,
        feature_size: int = 30,
        hops: int = 3,
    ):
        super().__init__()
        self._story_size = story_size
        self._query_size = query_size
        self._memory_key_size = memory_key_size
        self._memory_value_size = memory_value_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._feature_size = feature_size
        self.reader_feature_size = embedding_size

        self.register_buffer("_encoding", position_encoding(story_size, embedding_size))

        # frozen word-embedding matrix (real repo: loaded GloVe, trainable=False)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.requires_grad_(False)

        self.A = nn.Parameter(torch.empty(feature_size, self.reader_feature_size))
        self.A_mvalue = nn.Parameter(torch.empty(feature_size, self.reader_feature_size))
        self.A_mkey = nn.Parameter(torch.empty(feature_size, self.reader_feature_size))
        self.R = nn.Parameter(torch.empty(feature_size, feature_size))
        self.B = nn.Parameter(torch.empty(feature_size, score_range))
        self.logits_bias = nn.Parameter(torch.zeros(score_range))

        for p in (self.A, self.A_mvalue, self.A_mkey, self.R, self.B):
            nn.init.xavier_uniform_(p)

        self.dropout_rate = 0.2  # real repo default keep_prob=0.8 -> dropout rate 0.2

    def _key_addressing(
        self, mkeys: torch.Tensor, questions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mkeys: [batch, memory_key_size, reader_feature_size]
        questions: [batch, reader_feature_size]
        Returns (u_final [batch, feature_size], mem_attention_probs [batch, hops, memory_key_size])."""
        questions = F.dropout(questions, p=self.dropout_rate, training=self.training)
        # [batch, feature_size]
        u_o = questions @ self.A.t()
        u = [u_o]
        mem_attention_probs = []

        for _ in range(self._hops):
            mk_temp = F.dropout(mkeys, p=self.dropout_rate, training=self.training)
            # [batch, memory_size, feature_size]
            a_k = mk_temp @ self.A_mvalue.t()
            # [batch, 1, feature_size]
            u_expanded = u[-1].unsqueeze(1)
            # [batch, memory_size]
            dotted = (a_k * u_expanded).sum(dim=2)
            probs = F.softmax(dotted, dim=-1)
            mem_attention_probs.append(probs)
            # [batch, memory_size, 1]
            probs_expand = probs.unsqueeze(-1)
            mv_temp = mk_temp
            # [batch, memory_size, feature_size]
            a_v = mv_temp @ self.A_mkey.t()
            # [batch, feature_size]
            o_k = (probs_expand * a_v).sum(dim=1)
            u_k = F.relu(u[-1] @ self.R.t() + o_k @ self.R.t())
            u.append(u_k)

        return u[-1], torch.stack(mem_attention_probs, dim=1)

    def forward(
        self, query: torch.Tensor, memory_key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """query: [batch, query_size] token ids (the essay to score).
        memory_key: [batch, memory_value_size, story_size] token ids (one exemplar
        essay per score bucket, batched/repeated as in the real repo's
        `batched_memory = [memory] * (end-start)`)."""
        embedded_chars = self.embedding(query)  # [batch, query_size, embedding_size]
        mkeys_embedded_chars = self.embedding(
            memory_key
        )  # [batch, memory_size, story_size, embedding_size]

        # bow reader
        q_r = (embedded_chars * self._encoding[: embedded_chars.shape[1]]).sum(
            dim=1
        )  # [batch, embedding_size]
        doc_r = (mkeys_embedded_chars * self._encoding).sum(
            dim=2
        )  # [batch, memory_size, embedding_size]

        o, mem_attention_probs = self._key_addressing(doc_r, q_r)
        logits = o @ self.B + self.logits_bias
        probs = F.softmax(logits, dim=-1)
        return probs, mem_attention_probs


# ---- tiny build/example (architecture unmodified from the real repo's bow-reader path) ----


def build_kv_memn2n_aes():
    """Tiny MemN2N_KV (bow reader) for tracing. Architecture unmodified from the
    real repo's default hyperparameterization (hops=3, reader='bow')."""
    model = MemN2N_KV(
        vocab_size=64,
        query_size=10,
        story_size=10,
        memory_key_size=6,
        memory_value_size=6,
        embedding_size=16,
        score_range=6,
        feature_size=12,
        hops=3,
    )
    model.eval()
    return model


def example_input_kv_memn2n_aes():
    batch = 2
    vocab_size, query_size, story_size, memory_key_size = 64, 10, 10, 6
    query = torch.randint(0, vocab_size, (batch, query_size))
    memory_key = torch.randint(0, vocab_size, (batch, memory_key_size, story_size))
    return (query, memory_key)


MENAGERIE_ENTRIES = [
    (
        "MemN2N_KV_EssayGrading",
        build_kv_memn2n_aes,
        example_input_kv_memn2n_aes,
        2017,
        "ported-pytorch",
    ),
]
