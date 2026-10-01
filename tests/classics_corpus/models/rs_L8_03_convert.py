# SOURCE: vendored from jordiclive/Convert-PolyAI-Torch @ master
# (src/model.py, src/model_components.py, src/config.py, src/dataset.py)
#
# ConveRT (Conversational Representation from Transformer), Henderson et al.
# 2019 (https://arxiv.org/abs/1911.03688), PolyAI. This is a faithful
# open-source PyTorch reimplementation of the PolyAI dual-encoder
# architecture: subword + dual-modulo positional embedding, a stack of
# "SharedInnerBlock" relative-attention Transformer blocks with a learned
# circulant relative-position bias, a final multi-head self-attention
# reduction, and a 2-layer orthogonally-initialized FeedForward2 projection
# head (applied separately to context and reply) whose outputs are
# L2-normalized for response-selection similarity scoring.
#
# The queue candidate `davidalami/ConveRT` (community wrapper of the
# *original* PolyAI paper) is a pure TensorFlow-Hub client
# (conversational_sentence_encoder/vectorizers.py: tf.compat.v1 session +
# tfhub.Module against a frozen TF1 SavedModel) -- there is no traceable
# PyTorch nn.Module in that repo at all, and the official
# PolyAI-LDN/polyai-models repo ships no model code (README only). This
# jordiclive/Convert-PolyAI-Torch repo is a from-scratch-but-architecturally
# faithful PyTorch port of the same published architecture (matches the
# paper's SharedInnerBlock stack / FeedForward2 head / relative-attention
# bias design 1:1) and is REAL, existing, runnable repo code -- not a
# from-scratch approximation written here from the paper text.
#
# Vendoring notes (imports/config/wiring only, architecture untouched):
#   - `ConveRTModelConfig` is copied verbatim (a `typing.NamedTuple`); tiny
#     dims substituted for menagerie-scale tracing (kept the paper's
#     structural constraint `feed_forward2_hidden == num_embed_hidden *
#     num_attention_heads`, which is required because FeedForward2 consumes
#     the concatenated multi-head output of the final MultiheadAttention).
#   - `EncoderInputFeature` reduced from the original `@dataclass` (which
#     also carried an unused `input_lengths` field and a `pad_sequence`
#     helper method used only by the data pipeline) to the 3 fields the
#     model's forward pass actually reads: `input_ids`, `attention_mask`,
#     `position_ids`.
#   - `SingleContextConvert` (a `pytorch_lightning.LightningModule` in the
#     original, carrying training/optimizer/loss-function/lr-decay
#     scaffolding) is represented here by `DualEncoderConveRT`, a plain
#     `nn.Module` that performs the exact same forward computation
#     (`transformer_layers(x)` then per-branch `ff2_context`/`ff2_reply`,
#     see `SingleContextConvert.forward`/`training_step` in src/model.py) --
#     with the pytorch_lightning/optimizer/loss/data-loading machinery
#     stripped since it is orthogonal to the traceable architecture.

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


class ConveRTModelConfig(NamedTuple):
    num_embed_hidden: int = 16
    feed_forward1_hidden: int = 32
    feed_forward2_hidden: int = 32
    num_attention_project: int = 8
    vocab_size: int = 64
    num_encoder_layers: int = 2
    dropout_rate: float = 0.0
    n: int = 21
    relative_attns: list = [3, 5]
    num_attention_heads: int = 2
    token_sequence_truncation: int = 10


@dataclass
class EncoderInputFeature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


def circulant_mask(n: int, window: int) -> torch.Tensor:
    """Calculate the relative attention mask, calculated once when model instatiated, as a subset of this matrix
    will be used for a input length less than max.
    i,j represent relative token positions in this matrix and in the attention scores matrix,
     this mask enables attention scores to be set to 0 if further than the specified window length

        :param n: a fixed parameter set to be larger than largest max sequence length across batches
        :param window: [window length],
        :return relative attention mask
    """
    circulant_t = torch.zeros(n, n)
    offsets = [0] + [i for i in range(window + 1)] + [-i for i in range(window + 1)]
    if window >= n:
        return torch.ones(n, n)
    for offset in offsets:
        circulant_t.diagonal(offset=offset).copy_(torch.ones(n - abs(offset)))
    return circulant_t


class SubwordEmbedding(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.subword_embed = nn.Embedding(config.vocab_size, config.num_embed_hidden)
        self.m1_positional_embed = nn.Embedding(47, config.num_embed_hidden)
        self.m2_positional_embed = nn.Embedding(11, config.num_embed_hidden)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        subword_embed = self.subword_embed.forward(input_ids)
        m1_positional_embed = self.m1_positional_embed.forward(torch.fmod(position_ids, 47))
        m2_positional_embed = self.m2_positional_embed.forward(torch.fmod(position_ids, 11))
        embedding = subword_embed + m1_positional_embed + m2_positional_embed
        return embedding


class SelfAttention(nn.Module):
    """normal query, key, value based self attention but with relative attention functionality
    and a learnable bias encoding relative token position which is added to the attention scores before the softmax"""

    def __init__(self, config: ConveRTModelConfig, relative_attention: int):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.key = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.value = nn.Linear(config.num_embed_hidden, config.num_attention_project)

        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(config.num_attention_project, config.num_embed_hidden)
        self.bias = torch.nn.Parameter(torch.randn(config.n), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.bias.data.size(0))
        self.bias.data.uniform_(-stdv, stdv)
        self.relative_attention = relative_attention
        self.n = self.config.n
        self.half_n = self.n // 2
        self.register_buffer(
            "relative_mask",
            circulant_mask(config.token_sequence_truncation, self.relative_attention),
        )

    def forward(self, attn_input: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.T = attn_input.size()[1]
        _query = self.query.forward(attn_input)
        _key = self.key.forward(attn_input)
        _value = self.value.forward(attn_input)

        attention_scores = torch.matmul(_query, _key.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(self.config.num_attention_project)

        extended_attention_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        attention_scores = attention_scores + extended_attention_mask

        attention_scores = attention_scores.masked_fill(
            self.relative_mask.unsqueeze(0)[:, : self.T, : self.T] == 0, float("-inf")
        )

        ii, jj = torch.meshgrid(torch.arange(self.T), torch.arange(self.T))
        B_matrix = self.bias[self.n // 2 - ii + jj]

        attention_scores = attention_scores + B_matrix.unsqueeze(0)
        attention_scores = self.softmax(attention_scores)
        output = torch.matmul(attention_scores, _value)
        output = self.output_projection(output)
        return output


class FeedForward1(nn.Module):
    """feed-forward 1 is the standard FFN layer also used by Vaswani et al. (2017)"""

    def __init__(self, input_hidden: int, intermediate_hidden: int, dropout_rate: float = 0.0):
        super().__init__()
        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, input_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.linear_1(x))
        return self.linear_2(self.dropout(x))


class SharedInnerBlock(nn.Module):
    """Inner 'Transformer' block, repeated six times in the original paper with respective relative attentions
    [3, 5, 48, 48, 48, 48]"""

    def __init__(self, config: ConveRTModelConfig, relative_attn: int):
        super().__init__()
        self.config = config
        self.self_attention = SelfAttention(config, relative_attn)
        self.norm1 = LayerNorm(config.num_embed_hidden)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ff1 = FeedForward1(
            config.num_embed_hidden, config.feed_forward1_hidden, config.dropout_rate
        )
        self.norm2 = LayerNorm(config.num_embed_hidden)

    def forward(self, x: torch.Tensor, attention_mask: int) -> torch.Tensor:
        x = x + self.self_attention(x, attention_mask=attention_mask)
        x = self.norm2(x)
        x = x + self.ff1(x)
        return self.norm2(x)


class MultiheadAttention(nn.Module):
    """Standard non causal MHA, Half Hugging Face/Half Andrej Karpathy implementation,
    no need to mask as after previous layers"""

    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_attn_proj = config.num_embed_hidden * config.num_attention_heads
        self.attention_head_size = int(self.num_attn_proj / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.key = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.value = nn.Linear(config.num_embed_hidden, self.num_attn_proj)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = hidden_states.size()
        k = (
            self.key(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        q = (
            self.query(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        v = (
            self.value(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        y = attention_scores @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_attn_proj)
        return y


class TransformerLayers(nn.Module):
    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.config = config
        self.subword_embedding = SubwordEmbedding(config)
        self.transformer_layers = nn.ModuleList(
            [SharedInnerBlock(config, window) for window in config.relative_attns]
        )
        self.MHA = MultiheadAttention(config)

    def forward(self, encoder_input: EncoderInputFeature) -> torch.Tensor:
        input_ids = encoder_input.input_ids
        position_ids = encoder_input.position_ids
        attention_mask = encoder_input.attention_mask
        output = self.subword_embedding(input_ids, position_ids)
        for layer in self.transformer_layers:
            output = layer(output, attention_mask)
        output = self.MHA(output)
        return output


class FeedForward2(
    nn.Module
):  # params are not shared for context and reply, so two sets of weights
    """Fully-Connected residual projection head with orthogonal init and skip connections."""

    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.feed_forward2_hidden, config.feed_forward2_hidden)
        self.linear_2 = nn.Linear(config.feed_forward2_hidden, config.feed_forward2_hidden)
        self.norm1 = LayerNorm(config.feed_forward2_hidden)
        self.norm2 = LayerNorm(config.feed_forward2_hidden)
        self.final = nn.Linear(config.feed_forward2_hidden, config.num_embed_hidden)
        self.orthogonal_initialization()  # torch implementation works perfectly out the box

    def orthogonal_initialization(self):
        for layer in [self.linear_1, self.linear_2]:
            torch.nn.init.orthogonal_(layer.weight)

    def forward(self, x: torch.Tensor, attn_msk: torch.Tensor) -> torch.Tensor:
        sentence_lengths = attn_msk.sum(1)
        norms = 1 / torch.sqrt(sentence_lengths.double()).float()
        x = norms.unsqueeze(1) * torch.sum(x, dim=1)
        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))
        return F.normalize(self.final(x), dim=1, p=2)


class DualEncoderConveRT(nn.Module):
    """Single traceable wrapper matching `SingleContextConvert`'s forward computation
    (src/model.py): shared `transformer_layers` applied to context and reply, followed
    by separate `ff2_context`/`ff2_reply` projection heads. The pytorch_lightning
    training scaffolding (optimizer, loss function, lr-decay callback) from the
    original `SingleContextConvert` is intentionally omitted; the architecture itself
    is unchanged.
    """

    def __init__(self, config: ConveRTModelConfig):
        super().__init__()
        self.transformer_layers = TransformerLayers(config)
        self.ff2_context = FeedForward2(config)
        self.ff2_reply = FeedForward2(config)

    def forward(self, context: EncoderInputFeature, reply: EncoderInputFeature):
        rx = self.transformer_layers(context)
        ry = self.transformer_layers(reply)
        hx = self.ff2_context(rx, context.attention_mask)
        hy = self.ff2_reply(ry, reply.attention_mask)
        return hx, hy


MENAGERIE_ZOO = "vendored-pytorch"

_CONFIG = ConveRTModelConfig()


def build_convert():
    model = DualEncoderConveRT(_CONFIG)
    model.eval()
    return model


def example_input_convert():
    seq_len = _CONFIG.token_sequence_truncation
    batch = 2
    input_ids = torch.randint(1, _CONFIG.vocab_size, (batch, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
    attention_mask = torch.ones(batch, seq_len)
    context = EncoderInputFeature(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    )
    reply = EncoderInputFeature(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    )
    return (context, reply)


MENAGERIE_ENTRIES = [
    (
        "ConveRT (Conversational Representation from Transformer)",
        build_convert,
        example_input_convert,
        2019,
        "vendored-pytorch",
    ),
]
