# FAITHFUL PORT of moore3930/SlotRefine @ main (original framework: TensorFlow 1.x)
#
# https://github.com/moore3930/SlotRefine
# https://raw.githubusercontent.com/moore3930/SlotRefine/main/models.py
# https://raw.githubusercontent.com/moore3930/SlotRefine/main/thumt/models/transformer.py
# https://raw.githubusercontent.com/moore3930/SlotRefine/main/thumt/layers/attention.py
# https://raw.githubusercontent.com/moore3930/SlotRefine/main/thumt/layers/nn.py
#
# Qin et al. 2020 (EMNLP), "SlotRefine: A Fast Non-Autoregressive Model for
# Joint Intent Detection and Slot Filling". The original repo is TF1.x
# (`tensorflow.contrib`, static graphs, `tf.variable_scope`) built on top of a
# vendored copy of THUMT (Tsinghua's NMT toolkit); TF1.x + `tensorflow.contrib`
# cannot run in this base torch env, so the architecture is transcribed
# faithfully into self-contained torch below -- every mechanism from
# `NatSLU.create_model` (models.py:336-425), `transformer_encoder`
# (thumt/models/transformer.py:50-87), and THUMT's relative-position
# multi-head self-attention (thumt/layers/attention.py:378-479,
# `create_rpr`/`multiplicative_attention`) is preserved:
#
#   1. word embedding + slot-tag embedding, summed elementwise (not
#      concatenated) -- `inputs_emb + tags_emb` (models.py:359). The tag
#      embedding is what makes this "non-autoregressive two-pass iteration":
#      pass 1 feeds an all-"O" tag sequence, the model's *predicted* slot
#      tags for pass 1 (thresholded to B-* start tags only, see
#      `get_start_tags`, models.py:326-334) are then fed back in as
#      `input_tags` for pass 2, refining the slot predictions.
#   2. a learned CLS token prepended to the sequence (models.py:362-365),
#      whose transformer output is later split off and used for intent
#      classification.
#   3. bias/scale by sqrt(hidden_size) (`multiply_embedding_mode ==
#      "sqrt_depth"`) then a length-based padding mask multiply, plus a
#      single shared additive `bias` term (models.py:373-378).
#   4. `layer_preprocess="none"` / `layer_postprocess="layer_norm"` --
#      i.e. POST-norm transformer blocks (pre-norm branch is a no-op here),
#      matching `_layer_process` (thumt/models/transformer.py:18-24) called
#      with those two params.
#   5. multi-head self-attention with relative position representations
#      (Shaw et al. 2018, `max_relative_dis=16`, "relative" position mode):
#      learned `rpr_k`/`rpr_v` tables of size `[2*max_relative_dis+1, depth]`
#      indexed by clipped relative offset (`create_rpr`,
#      thumt/layers/attention.py:91-109), added into both the attention
#      logits and the attention-weighted values (`multiplicative_attention`,
#      thumt/layers/attention.py:314-375). No absolute/sinusoidal position
#      signal is added (`position_info_type == "relative"` skips
#      `add_timing_signal`).
#   6. position-wise FFN sublayer, ReLU, residual + post layer-norm
#      (`_ffn_layer`, thumt/models/transformer.py:33-47).
#   7. after the encoder stack, the output is split at the CLS position
#      (`tf.split(outputs, [1, len-1], 1)`, models.py:398): the CLS slice
#      goes through an `intent_proj` FFN -> intent logits; the remaining
#      per-token slice is concatenated with the (tiled) CLS hidden state and
#      goes through a `slot_proj` FFN -> per-token slot logits
#      (models.py:400-422). Both projections mask out the first two output
#      classes (PAD/UNK) with -1e10 so they are never predicted.
#
# The training loop, TF `tf.data`/`py_func` batching, THUMT NMT-model
# base class, checkpointing, and the ATIS/SNIPS tokenizer/evaluation
# scaffolding are infrastructure -- not part of the nn.Module architecture --
# and are not ported. Default hyperparameters below (hidden_size=32,
# filter_size=32, num_heads=8, num_encoder_layers=2, max_relative_dis=16,
# residual_dropout=0.1, attention_dropout=0.0) match the repo's argparse
# defaults (models.py:838-858).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_rpr_ids(length_q: int, length_kv: int, max_relative_dis: int, device) -> torch.Tensor:
    """Port of thumt/layers/attention.py `create_rpr` index computation."""
    idxs = torch.arange(length_kv, device=device).view(-1, 1)
    idys = torch.arange(length_kv, device=device).view(1, -1)
    ids = idxs - idys
    ids = ids + max_relative_dis
    ids = ids.clamp(min=0, max=2 * max_relative_dis)
    ids = ids[-length_q:, :]
    return ids


class RelativeMultiHeadSelfAttention(nn.Module):
    """Port of thumt multihead_attention (self-attention branch, with
    relative position representations) + multiplicative_attention."""

    def __init__(
        self, hidden_size: int, num_heads: int, max_relative_dis: int, attention_dropout: float
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_relative_dis = max_relative_dis
        self.attention_dropout = attention_dropout

        # qkv_transform: linear(queries, key_size*2 + value_size) with
        # key_size == value_size == hidden_size here (attention_key_channels
        # / attention_value_channels default to 0 -> falls back to hidden_size).
        self.qkv_transform = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.output_transform = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rpr_k = nn.Parameter(torch.zeros(2 * max_relative_dis + 1, self.head_dim))
        self.rpr_v = nn.Parameter(torch.zeros(2 * max_relative_dis + 1, self.head_dim))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [b, h, t, d]

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()  # [b, t, h, d]
        b, t, h, d = x.shape
        return x.view(b, t, h * d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # bias: [batch, 1, 1, length_kv] additive attention mask
        qkv = self.qkv_transform(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        length_q = q.shape[2]
        length_kv = k.shape[2]

        q = q * (self.head_dim**-0.5)

        rpr_ids = _create_rpr_ids(length_q, length_kv, self.max_relative_dis, x.device)
        rpr_k = self.rpr_k[rpr_ids]  # [length_q, length_kv, head_dim]
        rpr_v = self.rpr_v[rpr_ids]  # [length_q, length_kv, head_dim]

        logits_part1 = torch.matmul(q, k.transpose(-2, -1))  # [b, h, lq, lk]

        bs, hd = q.shape[0], q.shape[1]
        q_t = q.permute(2, 0, 1, 3).reshape(length_q, bs * hd, self.head_dim)
        logits_part2 = torch.matmul(q_t, rpr_k.transpose(-2, -1))  # [lq, bs*h, lk]
        logits_part2 = logits_part2.permute(1, 0, 2).reshape(bs, hd, length_q, length_kv)

        logits = logits_part1 + logits_part2
        if bias is not None:
            logits = logits + bias

        weights = F.softmax(logits, dim=-1)
        if self.training and self.attention_dropout > 0.0:
            weights = F.dropout(weights, p=self.attention_dropout)

        outputs_part1 = torch.matmul(weights, v)  # [b, h, lq, dv]

        w_t = weights.permute(2, 0, 1, 3).reshape(length_q, bs * hd, length_kv)
        outputs_part2 = torch.matmul(w_t, rpr_v)  # [lq, bs*h, dv]
        outputs_part2 = outputs_part2.permute(1, 0, 2).reshape(bs, hd, length_q, self.head_dim)

        outputs = outputs_part1 + outputs_part2

        combined = self._combine_heads(outputs)
        return self.output_transform(combined)


class FFNLayer(nn.Module):
    """Port of thumt _ffn_layer: linear -> relu -> linear."""

    def __init__(self, hidden_size: int, filter_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(hidden_size, filter_size, bias=True)
        self.output_layer = nn.Linear(filter_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.input_layer(x))
        return self.output_layer(hidden)


class TransformerEncoderLayer(nn.Module):
    """Port of one iteration of transformer_encoder's `for layer in range(...)`
    loop: self-attention (post-norm) + FFN (post-norm). layer_preprocess is
    "none" so pre-norm is a no-op; layer_postprocess is "layer_norm"."""

    def __init__(
        self,
        hidden_size: int,
        filter_size: int,
        num_heads: int,
        max_relative_dis: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.self_attention = RelativeMultiHeadSelfAttention(
            hidden_size, num_heads, max_relative_dis, attention_dropout
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FFNLayer(hidden_size, filter_size, hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.residual_dropout = residual_dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        y = self.self_attention(x, bias)
        if self.training and self.residual_dropout > 0.0:
            y = F.dropout(y, p=self.residual_dropout)
        x = x + y
        x = self.attn_layer_norm(x)

        y = self.ffn(x)
        if self.training and self.residual_dropout > 0.0:
            y = F.dropout(y, p=self.residual_dropout)
        x = x + y
        x = self.ffn_layer_norm(x)
        return x


class SlotRefine(nn.Module):
    """Port of `NatSLU.create_model` (models.py:336-425): word + slot-tag
    embedding -> CLS-prepended transformer encoder (relative-position
    self-attention, post-norm) -> split CLS for intent, remaining tokens
    (concatenated with tiled CLS state) for slot filling.

    A single `forward` call implements one refinement pass. The paper's
    "two-pass" inference (predict once with an all-"O" tag prior, extract
    predicted B-* start tags, feed them back in as `input_tags` for a second
    pass) is a *usage* pattern of this same module, not a distinct
    architecture -- callers can reproduce it by calling `forward` twice and
    feeding the first pass's slot argmax (thresholded to B-tag ids) back in
    as `input_tags`, exactly as `NatSLU.train_one_epoch`/`evaluation`/
    `inference` do via `get_start_tags` (models.py:326-334).
    """

    def __init__(
        self,
        vocab_size: int = 200,
        slot_size: int = 64,
        intent_size: int = 16,
        hidden_size: int = 32,
        filter_size: int = 32,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        max_relative_dis: int = 16,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.tag_embedding = nn.Embedding(slot_size, hidden_size)
        self.cls = nn.Parameter(torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size,
                    filter_size,
                    num_heads,
                    max_relative_dis,
                    attention_dropout,
                    residual_dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.intent_proj = FFNLayer(hidden_size, hidden_size, intent_size)
        # slot_proj input is [slot_output ; tiled intent_state] -> 2*hidden_size
        self.slot_proj = FFNLayer(hidden_size * 2, hidden_size, slot_size)

        self.residual_dropout = residual_dropout

    def forward(
        self, input_data: torch.Tensor, input_tags: torch.Tensor, sequence_length: torch.Tensor
    ) -> tuple:
        # input_data, input_tags: [batch, len_q] int64
        # sequence_length: [batch] int64, true (unpadded) token count per row
        batch_size, len_q = input_data.shape
        device = input_data.device

        inputs_emb = self.word_embedding(input_data)
        tags_emb = self.tag_embedding(input_tags)
        inputs = inputs_emb + tags_emb  # [batch, len_q, hidden]

        cls = self.cls.view(1, 1, -1).expand(batch_size, 1, -1)
        inputs = torch.cat([cls, inputs], dim=1)  # [batch, len_q+1, hidden]

        total_len = inputs.shape[1]
        positions = torch.arange(total_len, device=device).unsqueeze(0)
        src_mask = (positions < (sequence_length + 1).unsqueeze(1)).to(
            inputs.dtype
        )  # [batch, len_q+1]

        inputs = inputs * (self.hidden_size**-0.5)
        inputs = inputs * src_mask.unsqueeze(-1)
        encoder_input = inputs + self.bias

        # attention_bias(src_mask, "masking"): additive [-inf, 0] mask
        enc_attn_bias = (1.0 - src_mask) * -1e9
        enc_attn_bias = enc_attn_bias.view(batch_size, 1, 1, total_len)

        if self.training and self.residual_dropout > 0.0:
            encoder_input = F.dropout(encoder_input, p=self.residual_dropout)

        x = encoder_input
        for layer in self.encoder_layers:
            x = layer(x, enc_attn_bias)
        outputs = x  # layer_preprocess == "none" -> no final norm

        intent_state = outputs[:, :1, :]  # [batch, 1, hidden]
        slot_output = outputs[:, 1:, :]  # [batch, len_q, hidden]

        intent_output = self.intent_proj(intent_state)  # [batch, 1, intent_size]
        intent_mask = torch.zeros_like(intent_output, dtype=torch.bool)
        intent_mask[:, :, :2] = True
        intent_output = intent_output.masked_fill(intent_mask, -1e10)

        tiled_intent = intent_state.expand(-1, slot_output.shape[1], -1)
        slot_in = torch.cat([slot_output, tiled_intent], dim=-1)
        slot_output = self.slot_proj(slot_in)  # [batch, len_q, slot_size]
        slot_mask = torch.zeros_like(slot_output, dtype=torch.bool)
        slot_mask[:, :, :2] = True
        slot_output = slot_output.masked_fill(slot_mask, -1e10)

        return slot_output, intent_output


def build_slotrefine():
    return SlotRefine(
        vocab_size=64,
        slot_size=24,
        intent_size=12,
        hidden_size=32,
        filter_size=32,
        num_heads=8,
        num_encoder_layers=2,
        max_relative_dis=16,
        attention_dropout=0.0,
        residual_dropout=0.1,
    )


def example_input_slotrefine():
    batch_size, len_q = 2, 10
    input_data = torch.randint(2, 64, (batch_size, len_q))
    input_tags = torch.zeros(batch_size, len_q, dtype=torch.long)  # first-pass all-"O" prior
    sequence_length = torch.full((batch_size,), len_q, dtype=torch.long)
    return (input_data, input_tags, sequence_length)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("SlotRefine", "build_slotrefine", "example_input_slotrefine", 2020, "ported"),
]
