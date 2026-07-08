# FAITHFUL PORT of MiuLab/SlotGated-SLU @ master (original framework: TensorFlow 1.x)
#
# https://github.com/MiuLab/SlotGated-SLU
# https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/train.py
#
# Goo et al. 2018 (NAACL), "Slot-Gated Modeling for Joint Slot Filling and
# Intent Prediction". This is the paper's OFFICIAL repo -- the canonical
# source for both `cand_00125` (Slot-Gated SLU) and `cand_00126` (Slot-Gated
# Joint Model) queue rows, which both point at this same repository/model.
# The repo is TF1.x (`tf.contrib.rnn`, `tf.nn.bidirectional_dynamic_rnn`,
# `tf.placeholder`, static graph); TF1.x + `tensorflow.contrib` cannot run in
# this base torch env, so the architecture is transcribed faithfully into
# self-contained torch below. Every mechanism from `createModel`
# (train.py:81-174) is preserved:
#
#   1. word embedding -> bidirectional LSTM encoder (`layer_size` per
#      direction) producing per-token states `state_outputs` and a 4-way
#      concatenated final state (fw cell/hidden, bw cell/hidden) --
#      train.py:91-98.
#   2. slot attention (Bahdanau-style additive attention): a 1x1 conv
#      ("AttnW") over the per-token states plus a linear projection of the
#      per-token states themselves ("y"), summed, tanh'd, dotted with a
#      learned vector "AttnV", softmax'd over time, and used to pool
#      `hidden` (the per-token states) into a per-token context `slot_d`
#      (train.py:103-124). NOTE the query for this attention is the
#      per-token state itself (not a fixed query) -- every position gets its
#      own attention distribution over the whole sequence.
#   3. intent attention: the same additive-attention mechanism but with a
#      single global query (the concatenated final BiLSTM state) instead of
#      per-token queries, producing one pooled context `d` per example;
#      concatenated with the final state itself to give `intent_output`
#      (train.py:130-148, `add_final_state_to_intent=True` in the paper's
#      "full" model_type).
#   4. the slot gate (this paper's headline contribution): a linear
#      projection of `intent_output` ("intent_gate"), broadcast over time,
#      added to `slot_d`, tanh'd, dotted with a second learned vector
#      ("gateV") and summed over the hidden dim to get a scalar gate value
#      per token; that gate scales `slot_d` elementwise, and the gated
#      result is concatenated with the raw per-token BiLSTM states
#      (`slot_inputs`) before the final slot classifier (train.py:150-165).
#   5. two independent linear heads: `intent_proj` (global) and `slot_proj`
#      (per-token, from the gate-augmented representation) -- train.py:167-171.
#
# `model_type="full"` (both slot attention and the final-state contribution
# to intent enabled, i.e. `remove_slot_attn=False`,
# `add_final_state_to_intent=True`) is ported below, since that is the
# paper's headline configuration; the `intent_only` ablation
# (`remove_slot_attn=True`, which feeds raw BiLSTM states straight into the
# gate instead of slot-attention output) is exposed via the
# `remove_slot_attn` constructor flag for completeness but is not the
# default. The TF1.x `rnn_cell_impl._linear` helper is just a plain affine
# transform (`Wx + b`, with variable-length input concatenation); it is
# ported as `nn.Linear`. The training loop, gradient-splitting optimizer
# setup (separate slot/intent parameter groups), ATIS/SNIPS vocab/data
# pipeline, and checkpointing are infrastructure -- not part of the
# nn.Module architecture -- and are not ported. Default `layer_size=64`
# matches the repo's argparse default (train.py:17).

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Port of the slot/intent attention block shared by both branches in
    `createModel` (train.py:103-124 and 130-148): 1x1-conv-style projection
    of the memory + linear projection of the query, summed, tanh, dotted
    with a learned vector, softmax."""

    def __init__(self, attn_size: int):
        super().__init__()
        # AttnW: [1, 1, attn_size, attn_size] 1x1 conv == a per-position
        # linear projection applied to every timestep of `state_outputs`.
        self.attn_w = nn.Linear(attn_size, attn_size, bias=False)
        self.query_proj = nn.Linear(attn_size, attn_size, bias=True)
        self.attn_v = nn.Parameter(torch.zeros(attn_size))

    def forward(
        self, memory: torch.Tensor, query: torch.Tensor, per_token_query: bool
    ) -> torch.Tensor:
        # memory: [batch, seq_len, attn_size]
        hidden_features = self.attn_w(memory)  # [batch, seq_len, attn_size]

        if per_token_query:
            # slot attention: query == memory itself, one query per token.
            # y: [batch, seq_len, attn_size] -> broadcast against every
            # timestep of hidden_features: s[b, i, j] = sum_d v * tanh(hf[b,j,d] + y[b,i,d])
            y = self.query_proj(query)  # [batch, seq_len, attn_size]
            s = torch.tanh(
                hidden_features.unsqueeze(1) + y.unsqueeze(2)
            )  # [batch, tgt_i, src_j, attn]
            s = (self.attn_v * s).sum(-1)  # [batch, tgt_i, src_j]
            a = torch.softmax(s, dim=-1)  # softmax over src_j
            context = torch.einsum("bij,bjd->bid", a, memory)  # [batch, tgt_i, attn_size]
            return context
        else:
            # intent attention: a single global query -> one context vector.
            y = self.query_proj(query)  # [batch, attn_size]
            s = torch.tanh(hidden_features + y.unsqueeze(1))  # [batch, seq_len, attn_size]
            s = (self.attn_v * s).sum(-1)  # [batch, seq_len]
            a = torch.softmax(s, dim=-1)
            context = torch.einsum("bj,bjd->bd", a, memory)  # [batch, attn_size]
            return context


class SlotGatedSLU(nn.Module):
    """Port of `createModel` (train.py:81-174): BiLSTM encoder + slot/intent
    additive attention + the slot-gate mechanism + intent/slot classifiers."""

    def __init__(
        self,
        vocab_size: int = 200,
        slot_size: int = 40,
        intent_size: int = 20,
        layer_size: int = 64,
        remove_slot_attn: bool = False,
        add_final_state_to_intent: bool = True,
    ):
        super().__init__()
        self.layer_size = layer_size
        self.attn_size = layer_size * 2  # bidirectional concat width
        self.remove_slot_attn = remove_slot_attn
        self.add_final_state_to_intent = add_final_state_to_intent

        self.embedding = nn.Embedding(vocab_size, layer_size)
        self.encoder = nn.LSTM(layer_size, layer_size, batch_first=True, bidirectional=True)

        if not remove_slot_attn:
            self.slot_attn = AdditiveAttention(self.attn_size)
        self.intent_attn = AdditiveAttention(self.attn_size)

        intent_output_size = self.attn_size * 2 if add_final_state_to_intent else self.attn_size
        self.intent_gate_proj = nn.Linear(intent_output_size, self.attn_size, bias=True)
        self.gate_v = nn.Parameter(torch.zeros(self.attn_size))

        slot_input_size = self.attn_size * 2  # gated context concat raw per-token states
        self.intent_proj = nn.Linear(intent_output_size, intent_size, bias=True)
        self.slot_proj = nn.Linear(slot_input_size, slot_size, bias=True)

    def forward(self, input_data: torch.Tensor, sequence_length: torch.Tensor) -> tuple:
        # input_data: [batch, seq_len] int64 token ids
        # sequence_length: [batch] int64 true lengths (unused for masking
        # here, matching upstream: bidirectional_dynamic_rnn packs by
        # length but the attention itself is computed densely over the
        # full padded width, exactly as train.py does).
        inputs = self.embedding(input_data)  # [batch, seq_len, layer_size]

        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, sequence_length.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.encoder(packed)
        state_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=inputs.shape[1]
        )
        # state_outputs: [batch, seq_len, 2*layer_size]

        # final_state = concat(fw_c, fw_h, bw_c, bw_h) per train.py:96
        # (h_n/c_n layout: [num_directions, batch, hidden]); this is exactly
        # attn_size*2 == 4*layer_size wide by construction.
        final_state = torch.cat([c_n[0], h_n[0], c_n[1], h_n[1]], dim=-1)  # [batch, 4*layer_size]

        if not self.remove_slot_attn:
            slot_d = self.slot_attn(state_outputs, state_outputs, per_token_query=True)
            slot_inputs = state_outputs
        else:
            slot_d = state_outputs
            slot_inputs = state_outputs

        # intent attention uses the concatenated final BiLSTM state as its
        # single global query (train.py:129-148).
        intent_query = final_state[:, : self.attn_size]
        d = self.intent_attn(state_outputs, intent_query, per_token_query=False)

        if self.add_final_state_to_intent:
            intent_output = torch.cat([d, final_state[:, : self.attn_size]], dim=-1)
        else:
            intent_output = d

        intent_gate = self.intent_gate_proj(intent_output).unsqueeze(1)  # [batch, 1, attn_size]
        if not self.remove_slot_attn:
            slot_gate = self.gate_v * torch.tanh(slot_d + intent_gate)
        else:
            slot_gate = self.gate_v * torch.tanh(state_outputs + intent_gate)
        slot_gate = slot_gate.sum(-1, keepdim=True)  # [batch, seq_len, 1]
        if not self.remove_slot_attn:
            slot_gate = slot_d * slot_gate
        else:
            slot_gate = state_outputs * slot_gate

        slot_output = torch.cat([slot_gate, slot_inputs], dim=-1)  # [batch, seq_len, 2*attn_size]

        intent_logits = self.intent_proj(intent_output)  # [batch, intent_size]
        slot_logits = self.slot_proj(slot_output)  # [batch, seq_len, slot_size]

        return slot_logits, intent_logits


def build_slotgated_slu():
    return SlotGatedSLU(
        vocab_size=64,
        slot_size=24,
        intent_size=12,
        layer_size=16,
        remove_slot_attn=False,
        add_final_state_to_intent=True,
    )


def example_input_slotgated_slu():
    batch_size, seq_len = 2, 10
    input_data = torch.randint(1, 64, (batch_size, seq_len))
    sequence_length = torch.full((batch_size,), seq_len, dtype=torch.long)
    return (input_data, sequence_length)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("Slot-Gated SLU", "build_slotgated_slu", "example_input_slotgated_slu", 2018, "ported"),
]
