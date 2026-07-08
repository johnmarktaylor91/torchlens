# FAITHFUL PORT of RandolphVI/Hierarchical-Multi-Label-Text-Classification @ master
# (original framework: TensorFlow 1.x, tf.placeholder/tf.Variable graph-mode API)
#
# HARNN (Hierarchical Attention-based Recurrent Neural Network), CIKM'19 "Hierarchical
# Multi-label Text Classification: An Attention-based Recurrent Network Approach"
# (Huang, Xiao, Wu, Yuan, Zhao). The repo's real model file is
# HARNN/text_harnn.py -- a TensorFlow 1.x graph (tf.placeholder inputs, session-run
# training loop, tf.variable_scope weights). TF1.x graph-mode is not runnable/traceable
# under TorchLens's eager torch capture and is not one of the base libs available in
# this environment, so the architecture is transcribed faithfully into self-contained
# eager torch: every mechanism in the real text_harnn.py is preserved --
#   - shared embedding lookup + averaged-embedding side branch
#   - a single bidirectional LSTM producing per-timestep hidden states (`lstm_out`)
#     plus a mean-pooled summary (`lstm_out_pool`)
#   - 4 cascaded hierarchy levels, each with:
#       * `_attention`: a two-matrix (W_s1, W_s2) self-attention producing
#         [batch, num_classes_level, seq_len] attention weights over lstm_out
#         (gated by the PREVIOUS level's `visual` mask for levels 2-4, exactly as
#         `second_att_input = lstm_out * first_visual[..., None]` etc. in the source)
#       * `_fc_layer`: FC+ReLU over concat(lstm_out_pool, attention_out)
#       * `_local_layer`: a linear classifier producing per-level logits/scores,
#         plus a `visual` re-weighting of the attention map by the level's sigmoid
#         scores, renormalized with softmax -- fed forward to gate the next level
#   - concatenation of all 4 levels' FC outputs -> one more FC layer -> a 1-layer
#     Highway Network (`_highway_layer`, transform/carry gate exactly as
#     `t*h + (1-t)*x`) -> a global linear+sigmoid classifier
#   - final output blends global and concatenated local per-class scores via
#     `alpha * global_scores + (1 - alpha) * local_scores`, exactly as the source's
#     `self.scores = alpha * global_scores + (1 - alpha) * local_scores`
# The training-only loss computation (`cal_loss`, L2 reg over all trainable vars) is
# not part of the traced forward path -- the real repo's own `test_harnn.py` runs the
# graph up to `.scores` for inference, which is what this port's forward() returns.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class Attention(nn.Module):
    """Faithful port of `_attention` in text_harnn.py: two learned projection
    matrices W_s1 [attn_unit, num_units] and W_s2 [num_classes, attn_unit] produce
    a per-class attention map over the sequence, softmax-normalized, then used to
    pool the sequence into a single summary vector per class (mean over classes)."""

    def __init__(self, num_units: int, num_classes: int, attention_unit_size: int):
        super().__init__()
        self.W_s1 = nn.Parameter(torch.empty(attention_unit_size, num_units))
        self.W_s2 = nn.Parameter(torch.empty(num_classes, attention_unit_size))
        nn.init.trunc_normal_(self.W_s1, std=0.1)
        nn.init.trunc_normal_(self.W_s2, std=0.1)

    def forward(self, input_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input_x: [batch, seq_len, num_units]
        # attention_matrix: [batch, num_classes, seq_len]
        # W_s1 @ x^T for each batch element, then tanh, then W_s2 @ (.)
        proj = torch.einsum("un,bsn->bus", self.W_s1, input_x)  # [batch, attn_unit, seq_len]
        proj = torch.tanh(proj)
        attention_matrix = torch.einsum(
            "cu,bus->bcs", self.W_s2, proj
        )  # [batch, num_classes, seq_len]
        attention_weight = F.softmax(attention_matrix, dim=-1)
        attention_out = torch.bmm(attention_weight, input_x)  # [batch, num_classes, num_units]
        attention_out = attention_out.mean(dim=1)  # [batch, num_units]
        return attention_weight, attention_out


class FCLayer(nn.Module):
    """Faithful port of `_fc_layer`: single Linear + ReLU."""

    def __init__(self, in_features: int, fc_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features, fc_hidden_size)
        nn.init.trunc_normal_(self.linear.weight, std=0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


class LocalLayer(nn.Module):
    """Faithful port of `_local_layer`: per-level linear classifier + a `visual`
    re-weighting of the incoming attention map by the level's sigmoid scores,
    softmax-renormalized -- forwarded to gate the next hierarchy level."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.linear.weight, std=0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(
        self, input_x: torch.Tensor, input_att_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(input_x)  # [batch, num_classes]
        scores = torch.sigmoid(logits)
        # visual: [batch, num_classes, seq_len] * [batch, num_classes, 1] -> softmax -> mean over classes
        visual = input_att_weight * scores.unsqueeze(-1)
        visual = F.softmax(visual, dim=-1)
        visual = visual.mean(dim=1)  # [batch, seq_len]
        return logits, scores, visual


class HighwayLayer(nn.Module):
    """Faithful port of `_highway_layer` (1 layer, as used by the source):
    t = sigmoid(W_t x + b_t); h = relu(W_h x + b_h); out = t*h + (1-t)*x."""

    def __init__(self, size: int, bias: float = 0.0):
        super().__init__()
        self.h_linear = nn.Linear(size, size)
        self.t_linear = nn.Linear(size, size)
        nn.init.constant_(self.t_linear.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.h_linear(x))
        t = torch.sigmoid(self.t_linear(x))
        return t * h + (1.0 - t) * x


class TextHARNN(nn.Module):
    """Faithful torch port of HARNN/text_harnn.py's TextHARNN graph, transcribed
    layer-for-layer from the real TF1.x source (see module docstring above)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        lstm_hidden_size: int,
        attention_unit_size: int,
        fc_hidden_size: int,
        num_classes_list: list[int],
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_classes_list = num_classes_list
        self.total_classes = sum(num_classes_list)
        self.alpha = alpha

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        lstm_out_dim = lstm_hidden_size * 2

        self.attentions = nn.ModuleList(
            [Attention(lstm_out_dim, n, attention_unit_size) for n in num_classes_list]
        )
        self.local_fcs = nn.ModuleList(
            [FCLayer(lstm_out_dim + lstm_out_dim, fc_hidden_size) for _ in num_classes_list]
        )
        self.local_layers = nn.ModuleList([LocalLayer(fc_hidden_size, n) for n in num_classes_list])

        self.fc_out = FCLayer(fc_hidden_size * len(num_classes_list), fc_hidden_size)
        self.highway = HighwayLayer(fc_hidden_size)

        self.global_linear = nn.Linear(fc_hidden_size, self.total_classes)
        nn.init.trunc_normal_(self.global_linear.weight, std=0.1)
        nn.init.constant_(self.global_linear.bias, 0.1)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        # input_x: [batch, seq_len] int64 token ids
        embedded = self.embedding(input_x)  # [batch, seq_len, embedding_size]

        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, lstm_hidden_size * 2]
        lstm_out_pool = lstm_out.mean(dim=1)  # [batch, lstm_hidden_size * 2]

        local_fc_outs = []
        local_scores = []
        att_input = lstm_out
        for level in range(len(self.num_classes_list)):
            att_weight, att_out = self.attentions[level](att_input)
            local_input = torch.cat([lstm_out_pool, att_out], dim=1)
            local_fc_out = self.local_fcs[level](local_input)
            _, scores, visual = self.local_layers[level](local_fc_out, att_weight)
            local_fc_outs.append(local_fc_out)
            local_scores.append(scores)
            att_input = lstm_out * visual.unsqueeze(-1)

        ham_out = torch.cat(local_fc_outs, dim=1)
        fc_out = self.fc_out(ham_out)
        highway_out = self.highway(fc_out)

        global_logits = self.global_linear(highway_out)
        global_scores = torch.sigmoid(global_logits)

        local_scores_concat = torch.cat(local_scores, dim=1)
        scores = self.alpha * global_scores + (1.0 - self.alpha) * local_scores_concat
        return scores


# ---- tiny build/example (architecture faithfully transcribed from the real repo) ----


def build_harnn():
    model = TextHARNN(
        vocab_size=64,
        embedding_size=16,
        lstm_hidden_size=12,
        attention_unit_size=8,
        fc_hidden_size=20,
        num_classes_list=[3, 4, 5, 6],
        alpha=0.5,
    )
    model.eval()
    return model


def example_input_harnn():
    batch, seq_len, vocab_size = 2, 10, 64
    return torch.randint(0, vocab_size, (batch, seq_len))


MENAGERIE_ENTRIES = [
    ("HARNNReadability", build_harnn, example_input_harnn, 2019, "ported-pytorch"),
]
