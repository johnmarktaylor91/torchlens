# FAITHFUL PORT of https://github.com/tech-srl/code2vec @ master (original framework: TF1-era
# Keras, `tensorflow.python.keras` / `tf.compat.v1.disable_eager_execution()`)
#
# code2vec (Alon, Zilberstein, Levy & Yahav, POPL 2019) represents a code snippet as a
# bag of AST path-contexts (source-terminal token, path, target-terminal token), embeds
# each of the three parts, concatenates + denses each context to a fixed-size vector,
# then aggregates the per-context vectors into one code vector via a learned linear
# *attention* mechanism (see repo file `keras_attention_layer.py`, class
# `AttentionLayer`, and `keras_model.py::Code2VecModel._create_keras_model`). We port
# the real architecture (embeddings -> per-context dense -> attention pooling -> softmax
# target-word head) faithfully layer-for-layer; only the framework changes from
# TF1-era Keras custom layers to torch nn.Module (attention math ported one-for-one from
# `keras_attention_layer.py`'s `call()`: dot(x, attention_param) -> mask via log-add ->
# softmax over the context axis -> weighted sum). We can't vendor the TF1 code directly:
# it needs `tensorflow.python.keras` internals and `tf.compat.v1` graph-mode session
# execution, which are not part of this environment's base-lib set, and the repo has no
# TF2/Keras-3/torch reimplementation.

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Port of keras_attention_layer.py::AttentionLayer.

    Learns a single (input_dim, 1) attention parameter vector; scores each context by
    dot product with it, adds a log-mask for padded contexts, softmaxes over the
    context axis, then returns the mask-weighted sum of contexts (the "code vector")
    plus the raw attention weights.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention_param = nn.Parameter(torch.empty(input_dim, 1).uniform_(-0.05, 0.05))

    def forward(self, inputs, mask=None):
        # inputs: (batch, input_length, input_dim)
        attention_weights = inputs @ self.attention_param  # (batch, input_length, 1)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(2)  # (batch, input_length, 1)
            attention_weights = attention_weights + torch.log(mask.clamp_min(1e-45))

        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch, input_length, 1)
        result = (inputs * attention_weights).sum(dim=1)  # (batch, input_dim)
        return result, attention_weights


class Code2Vec(nn.Module):
    """Port of keras_model.py::Code2VecModel._create_keras_model (train-graph path,
    dropping the eval-only top-k word lookup head which is pure post-processing over
    the same target_index logits)."""

    def __init__(
        self,
        token_vocab_size,
        path_vocab_size,
        target_vocab_size,
        token_embed_size=32,
        path_embed_size=32,
        code_vector_size=32,
        dropout_keep_rate=0.75,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(token_vocab_size, token_embed_size)
        self.path_embedding = nn.Embedding(path_vocab_size, path_embed_size)

        context_dim = 2 * token_embed_size + path_embed_size
        self.dropout = nn.Dropout(1 - dropout_keep_rate)
        self.context_dense = nn.Linear(context_dim, code_vector_size, bias=False)
        self.attention = AttentionLayer(code_vector_size)
        self.target_index = nn.Linear(code_vector_size, target_vocab_size, bias=False)

    def forward(self, inputs):
        path_source_token_input, path_input, path_target_token_input, context_valid_mask = inputs

        paths_embedded = self.path_embedding(path_input)
        path_source_token_embedded = self.token_embedding(path_source_token_input)
        path_target_token_embedded = self.token_embedding(path_target_token_input)

        context_embedded = torch.cat(
            [path_source_token_embedded, paths_embedded, path_target_token_embedded], dim=-1
        )
        context_embedded = self.dropout(context_embedded)

        context_after_dense = torch.tanh(self.context_dense(context_embedded))

        code_vectors, attention_weights = self.attention(
            context_after_dense, mask=context_valid_mask
        )

        target_logits = torch.softmax(self.target_index(code_vectors), dim=-1)
        return target_logits


MENAGERIE_ZOO = "ported-pytorch"

_TOKEN_VOCAB = 64
_PATH_VOCAB = 48
_TARGET_VOCAB = 40
_MAX_CONTEXTS = 20


def build_code2vec():
    model = Code2Vec(
        token_vocab_size=_TOKEN_VOCAB,
        path_vocab_size=_PATH_VOCAB,
        target_vocab_size=_TARGET_VOCAB,
        token_embed_size=16,
        path_embed_size=16,
        code_vector_size=16,
    )
    model.eval()
    return model


def example_input_code2vec():
    batch = 4
    path_source_token_input = torch.randint(0, _TOKEN_VOCAB, (batch, _MAX_CONTEXTS))
    path_input = torch.randint(0, _PATH_VOCAB, (batch, _MAX_CONTEXTS))
    path_target_token_input = torch.randint(0, _TOKEN_VOCAB, (batch, _MAX_CONTEXTS))
    context_valid_mask = torch.ones(batch, _MAX_CONTEXTS)
    return (path_source_token_input, path_input, path_target_token_input, context_valid_mask)


MENAGERIE_ENTRIES = [
    ("code2vec", build_code2vec, example_input_code2vec, 2019, "MENAGERIE_ZOO"),
]
