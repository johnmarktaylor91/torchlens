# FAITHFUL PORT of ZephyrChenzf/SF-ID-Network-For-NLU @ master (original
# framework: TensorFlow 1.x / tf.contrib)
#
# https://github.com/ZephyrChenzf/SF-ID-Network-For-NLU
# https://raw.githubusercontent.com/ZephyrChenzf/SF-ID-Network-For-NLU/master/train.py
#
# Zhu Su et al. 2019 "A Novel Bi-directional Interrelated Model for Joint
# Intent Detection and Slot Filling" (SF-ID Network, ACL 2019). The official
# repo is TF1.x (`tensorflow.contrib.rnn`, `core_rnn_cell._linear`,
# `tf.nn.bidirectional_dynamic_rnn`) -- `tensorflow.contrib` was removed in
# TF2 and is not installable in this environment, so this is a FAITHFUL PORT
# rather than a vendor. No PyTorch reimplementation of this exact
# architecture (with its distinctive SF-ID iterative-refinement subnet
# stack, as opposed to the earlier, architecturally different slot-gated
# predecessor) could be located after a GitHub code/repo search
# (`bo-ke/cybo`'s `cybo/models/sf_id.py` is also TensorFlow/Keras, not
# PyTorch).
#
# Every mechanism of the real `createModel` (train.py:88-234, default
# `priority_order='slot_first'`, `model_type='full'` i.e. `remove_slot_attn
# =False`, `use_crf=False`, `iteration_num=1` -- the repo's own defaults for
# the snips/atis "full" configuration) is transcribed with exact tensor
# algebra, worked out shape-by-shape from the TF ops:
#   - `cell_fw`/`cell_bw` = `BasicLSTMCell` + `bidirectional_dynamic_rnn`,
#     `state_outputs` = concat of both directions' per-step hidden states
#     (train.py:90-107) -> `nn.LSTM(bidirectional=True, batch_first=True)`.
#     `final_state` = concat of `[fwd_c, fwd_h, bwd_c, bwd_h]`
#     (`LSTMStateTuple(c, h)` per direction, train.py:106) -> reproduced in
#     the same concat order from `nn.LSTM`'s `(h_n, c_n)` outputs.
#   - `slot_attn` (train.py:113-132) is a full self-attention over the
#     sequence: for every query position i, `hidden_features[j]` is a linear
#     projection of `state_outputs[j]` (the 1x1 conv2d "AttnW" kernel over
#     the channel axis is algebraically an nn.Linear at each position) and
#     `y[i]` is a *separate* linear projection ("_linear", `core_rnn_cell.
#     _linear`) of `state_outputs[i]`; the additive Bahdanau score
#     `s[i,j] = sum_h(v * tanh(hidden_features[j] + y[i]))` is softmaxed
#     over j and used to pool `slot_d[i] = sum_j a[i,j] * state_outputs[j]`
#     -> ported as `SlotSelfAttn` below, an explicit (B,T,T,H) score tensor
#     matching the TF broadcast exactly (`hidden`=expand_dims(...,1)` is the
#     key axis, `y`=expand_dims(...,2)` is the query axis in the TF code).
#   - `intent_attn` (train.py:140-155) is the same additive-attention form,
#     but with the query fixed to `final_state` (not per-position) -> ported
#     as `IntentAttn`.
#   - the `priority_order == 'slot_first'` SF-ID subnet stack (the paper's
#     core contribution, train.py:192-224): `slot_subnet` computes a
#     per-position gate from `r_intent` (`intent_gate = linear(r_intent)`,
#     broadcast over T) and reinforces `slot_d` via elementwise
#     `v1 * tanh(slot_d + intent_gate)` gating, producing
#     `slot_reinforce_state = slot_d * relation_factor`; `intent_subnet`
#     re-attends `state_outputs` (query, via conv kernel "W2") against the
#     reinforced slot state (key, via conv kernel "W1") plus a bias term,
#     producing a refined `r_intent = r + intent_context_states` that is
#     concatenated with `intent_input` (`final_state`) for classification
#     -> ported as `SlotSubnet`/`IntentSubnet`, iterated `iteration_num`
#     times exactly as in the TF loop (the `intent_first` branch,
#     train.py:157-190, is algebraically the mirror image and is omitted:
#     not exercised, since `priority_order='slot_first'` is the default).
#   - final `intent_proj`/`slot_proj` linear heads over
#     `concat([r_intent, intent_input])` / `concat([slot_reinforce_vector,
#     slot_inputs])` (`core_rnn_cell._linear`, train.py:226-229) ->
#     `nn.Linear`.
#   - CRF slot decoding (`use_crf=True` branch), the `DropoutWrapper`s
#     (training-time regularization, not architecture), the embedding
#     lookup table swap for pretrained GloVe (`arg.embedding_path`), and all
#     TF1 session/optimizer/checkpoint/gradient-clipping plumbing
#     (train.py:236-471) are dropped -- they configure *how* the graph is
#     trained/fed, not what tensors the forward pass over a single batch of
#     token ids computes, which is what TorchLens captures.

import torch
import torch.nn as nn


class SlotSelfAttn(nn.Module):
    """Faithful port of the TF `slot_attn` block (train.py:113-132): a full
    self-attention over the sequence producing a per-position context
    `slot_d[i] = sum_j softmax_j(v . tanh(W_key(x_j) + W_query(x_i))) * x_j`.
    """

    def __init__(self, attn_size):
        super().__init__()
        self.attn_w = nn.Linear(
            attn_size, attn_size
        )  # conv2d "AttnW" (1x1 kernel) -> per-position key proj
        self.linear_y = nn.Linear(
            attn_size, attn_size
        )  # core_rnn_cell._linear -> per-position query proj
        self.v = nn.Parameter(torch.randn(attn_size) * 0.1)  # "AttnV"

    def forward(self, state_outputs):
        # state_outputs: (B, T, H)
        hidden_features = self.attn_w(
            state_outputs
        )  # (B, T, H), keyed by j -> unsqueeze(1) below: (B,1,T,H)
        y = self.linear_y(state_outputs)  # (B, T, H), keyed by i -> unsqueeze(2) below: (B,T,1,H)
        s = (self.v * torch.tanh(hidden_features.unsqueeze(1) + y.unsqueeze(2))).sum(
            dim=3
        )  # (B, T_i, T_j)
        a = torch.softmax(s, dim=2).unsqueeze(-1)  # (B, T_i, T_j, 1)
        slot_d = (a * state_outputs.unsqueeze(1)).sum(dim=2)  # (B, T_i, H)
        return slot_d


class IntentAttn(nn.Module):
    """Faithful port of the TF `intent_attn` block (train.py:140-155):
    additive attention over `state_outputs` with the query fixed to
    `final_state` (not per-position). `final_state` is the 4-way
    concatenation of both LSTM directions' (c, h) states (`4*layer_size`
    wide -- see `SFIDNetwork.forward`), a *different* width from
    `attn_size` (`2*layer_size`, `state_outputs`'s channel width); TF's
    `core_rnn_cell._linear(intent_input, attn_size, True)` (train.py:147)
    projects from whatever width `intent_input` has down to `attn_size`,
    so `linear_y` here takes `final_state_size` in, not `attn_size`.
    """

    def __init__(self, attn_size, final_state_size):
        super().__init__()
        self.attn_w = nn.Linear(attn_size, attn_size)  # conv2d "AttnW"
        self.linear_y = nn.Linear(
            final_state_size, attn_size
        )  # core_rnn_cell._linear over final_state
        self.v = nn.Parameter(torch.randn(attn_size) * 0.1)  # "AttnV"

    def forward(self, state_outputs, final_state):
        # state_outputs: (B, T, H); final_state: (B, H)
        hidden_features = self.attn_w(state_outputs)  # (B, T, H)
        y = self.linear_y(final_state).unsqueeze(1)  # (B, 1, H)
        s = (self.v * torch.tanh(hidden_features + y)).sum(dim=2)  # (B, T)
        a = torch.softmax(s, dim=1).unsqueeze(-1)  # (B, T, 1)
        d = (a * state_outputs).sum(dim=1)  # (B, H)
        return d


class SlotSubnet(nn.Module):
    """Faithful port of TF `slot_subnet` (train.py:194-204, the
    `priority_order == 'slot_first'` branch): an intent-gated relation
    factor elementwise-reinforces the slot self-attention context `slot_d`.
    """

    def __init__(self, attn_size):
        super().__init__()
        self.intent_gate = nn.Linear(attn_size, attn_size)  # core_rnn_cell._linear over r_intent
        self.gate_v = nn.Parameter(torch.randn(attn_size) * 0.1)  # "gateV"

    def forward(self, slot_d, r_intent):
        # slot_d: (B, T, H); r_intent: (B, H)
        intent_gate = self.intent_gate(r_intent).unsqueeze(1)  # (B, 1, H)
        relation_factor = (self.gate_v * torch.tanh(slot_d + intent_gate)).sum(
            dim=2, keepdim=True
        )  # (B, T, 1)
        slot_reinforce_state = slot_d * relation_factor  # (B, T, H)
        return slot_reinforce_state


class IntentSubnet(nn.Module):
    """Faithful port of TF `intent_subnet` (train.py:206-224, the
    `priority_order == 'slot_first'` branch): a second additive attention
    over `state_outputs` (query, via conv kernel "W2") against the
    reinforced slot state (key, via conv kernel "W1") plus a bias term.
    """

    def __init__(self, attn_size):
        super().__init__()
        self.w1 = nn.Linear(
            attn_size, attn_size, bias=False
        )  # conv2d "W1" over slot_reinforce_state
        self.w2 = nn.Linear(attn_size, attn_size, bias=False)  # conv2d "W2" over state_outputs
        self.v = nn.Parameter(torch.randn(attn_size) * 0.1)  # "AttnV"
        self.bias = nn.Parameter(torch.zeros(attn_size))  # "Bias"

    def forward(self, state_outputs, slot_reinforce_state, intent_context_states):
        # state_outputs, slot_reinforce_state: (B, T, H); intent_context_states: (B, H)
        slot_features = self.w1(slot_reinforce_state)  # (B, T, H)
        hidden_features = self.w2(state_outputs)  # (B, T, H)
        s = (self.v * torch.tanh(hidden_features + slot_features + self.bias)).sum(dim=2)  # (B, T)
        a = torch.softmax(s, dim=1).unsqueeze(-1)  # (B, T, 1)
        r = (a * slot_reinforce_state).sum(dim=1)  # (B, H)
        r_intent = r + intent_context_states
        return r_intent


class SFIDNetwork(nn.Module):
    """Faithful torch port of `createModel` (train.py:88-234): a
    bidirectional-LSTM token encoder, dual (slot self-attention + intent
    attention) blocks, and an iterative SF-ID subnet stack (slot-first
    priority order) that lets slot and intent predictions mutually reinforce
    each other before the final classification heads.
    """

    def __init__(self, vocab_size, slot_size, intent_size, layer_size=64, iteration_num=1):
        super().__init__()
        self.layer_size = layer_size
        self.iteration_num = iteration_num
        attn_size = layer_size * 2  # bidirectional concat width == state_shape[2] in TF

        final_state_size = (
            layer_size * 4
        )  # concat([fwd_c, fwd_h, bwd_c, bwd_h]), each layer_size wide

        self.embedding = nn.Embedding(vocab_size, layer_size)
        self.bi_lstm = nn.LSTM(layer_size, layer_size, batch_first=True, bidirectional=True)

        self.slot_attn = SlotSelfAttn(attn_size)
        self.intent_attn = IntentAttn(attn_size, final_state_size)

        self.slot_subnets = nn.ModuleList([SlotSubnet(attn_size) for _ in range(iteration_num)])
        self.intent_subnets = nn.ModuleList([IntentSubnet(attn_size) for _ in range(iteration_num)])

        # intent_output = concat([r_intent (attn_size), intent_input==final_state (final_state_size)])
        self.intent_proj = nn.Linear(attn_size + final_state_size, intent_size)
        self.slot_proj = nn.Linear(
            attn_size * 2, slot_size
        )  # concat([slot_reinforce_vector, slot_inputs])

    def forward(self, input_data):
        # input_data: (B, T) token ids
        inputs = self.embedding(input_data)  # (B, T, layer_size)
        state_outputs, (h_n, c_n) = self.bi_lstm(inputs)  # state_outputs: (B, T, 2*layer_size)
        # h_n, c_n: (2, B, layer_size); TF concat order = [fwd_c, fwd_h, bwd_c, bwd_h]
        # (LSTMStateTuple(c, h) per direction, forward then backward -- train.py:106)
        final_state = torch.cat(
            [c_n[0], h_n[0], c_n[1], h_n[1]], dim=1
        )  # (B, 4*layer_size) == (B, attn_size)

        slot_inputs = state_outputs  # (B, T, attn_size), kept for the final slot_output concat
        slot_d = self.slot_attn(state_outputs)  # (B, T, attn_size)

        intent_input = final_state  # (B, attn_size)
        intent_context_states = self.intent_attn(state_outputs, final_state)  # (B, attn_size)
        r_intent = intent_context_states

        slot_reinforce_state = slot_d
        for n in range(self.iteration_num):
            slot_reinforce_state = self.slot_subnets[n](slot_d, r_intent)  # (B, T, attn_size)
            r_intent = self.intent_subnets[n](
                state_outputs, slot_reinforce_state, intent_context_states
            )  # (B, attn_size)

        intent_output = torch.cat([r_intent, intent_input], dim=1)  # (B, 2*attn_size)
        slot_output = torch.cat([slot_reinforce_state, slot_inputs], dim=2)  # (B, T, 2*attn_size)

        intent_logits = self.intent_proj(intent_output)  # (B, intent_size)
        slot_logits = self.slot_proj(slot_output)  # (B, T, slot_size)
        return slot_logits, intent_logits


def build_sf_id_network():
    vocab_size = 60
    slot_size = 12
    intent_size = 8
    layer_size = 16
    iteration_num = 1
    return SFIDNetwork(
        vocab_size, slot_size, intent_size, layer_size=layer_size, iteration_num=iteration_num
    )


def example_input_sf_id_network():
    batch_size = 2
    seq_len = 10
    vocab_size = 60
    return torch.randint(0, vocab_size, (batch_size, seq_len))


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SF-ID Network (joint intent detection + slot filling)",
        "build_sf_id_network",
        "example_input_sf_id_network",
        2019,
        "ported",
    ),
]
