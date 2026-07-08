# FAITHFUL PORT of RandolphVI/Hierarchical-Multi-Label-Text-Classification @ master
# (original framework: TensorFlow 1.x)
# https://raw.githubusercontent.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification/master/HARNN/text_harnn.py
# https://raw.githubusercontent.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification/master/utils/param_parser.py
#
# Huang, Xie, Wang, Peng, Chen, Xiao 2019 (SIGIR) "Hierarchical Multi-Label Text
# Classification: An Attention-Based Recurrent Network Approach" -- HARNN (Hierarchical
# Attention-based Recurrent Neural Network), applied to patent classification. A
# bidirectional LSTM encodes the token sequence; at each of the 4 hierarchy levels
# (`first`/`second`/`third`/`fourth`, coarse -> fine label taxonomy depth), a level-specific
# attention layer (`_attention`, a `tanh(W_s1 x) -> W_s2` two-matrix attention identical to
# the Lin et al. 2017 "A Structured Self-Attentive Sentence Embedding" mechanism) scores
# every timestep against that level's classes; each level's "visual" (softmax-normalized,
# per-class score-weighted attention map, `_local_layer`) gates which timesteps the NEXT,
# finer level's attention is allowed to attend to (`second_att_input = lstm_out *
# first_visual`, etc.) -- the hierarchy-aware attention propagation that gives the paper its
# name. Each level also produces its own local per-class sigmoid logits (`_local_layer`).
# The four levels' FC outputs are concatenated (`ham_out`), pushed through one more FC layer
# and a Highway layer (`_highway_layer`, Srivastava et al. 2015), then dropout, then a
# global fully-connected classifier over the FULL flat label set. Final scores are an
# alpha-weighted sum of the global sigmoid scores and the concatenation of all 4 levels'
# local sigmoid scores.
#
# `TextHARNN.__init__`'s TF1 graph-construction body (`_attention`, `_fc_layer`,
# `_local_layer`, `_linear`, `_highway_layer`, the Bi-LSTM block, and the 4-level cascade +
# concat + FC + highway + global-logits + score-blend) is transcribed op-for-op into
# `nn.Module` methods below: `tf.nn.bidirectional_dynamic_rnn` -> `nn.LSTM(bidirectional=True)`;
# `tf.map_fn(matmul(W_s1, x^T))` batched attention -> `torch.einsum`/`torch.bmm` batched
# matmul (same per-example `W_s1 @ x^T` then `tanh` then `W_s2 @ (.)` computation, just
# vectorized across the batch instead of TF's per-example `map_fn`); `tf.nn.xw_plus_b` ->
# `nn.Linear`; `_highway_layer`'s `t*h + (1-t)*x` gate is copied verbatim; the loss's
# `cal_loss` (per-level `BCEWithLogitsLoss` summed over classes, mean over batch) and the
# final `local_losses + global_losses + l2_losses` combination are reproduced as
# `compute_loss`. No architectural change: same 4-level hierarchy-propagated attention,
# same highway/global/local score blend, same loss composition. Hyperparameter names/values
# (`pad_seq_len`, `embedding_dim=100`, `lstm_dim=256`, `attention_dim=200`, `fc_dim=512`,
# `num_classes_list=[9,128,661,8364]`, `total_classes=9162`, `alpha=0.5`,
# `dropout_rate=0.5`) mirror `utils/param_parser.py`'s real argparse defaults; the staging
# build uses tiny values of the same fields for a fast trace.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Structured self-attention (Lin et al. 2017) scoring every timestep against
    `num_classes` learned queries -- transcribed from TextHARNN's `_attention` closure."""

    def __init__(self, num_units, attention_unit_size, num_classes):
        super().__init__()
        self.W_s1 = nn.Parameter(torch.empty(attention_unit_size, num_units))
        self.W_s2 = nn.Parameter(torch.empty(num_classes, attention_unit_size))
        nn.init.trunc_normal_(self.W_s1, std=0.1)
        nn.init.trunc_normal_(self.W_s2, std=0.1)

    def forward(self, input_x):
        """
        input_x: [batch, seq_len, num_units]
        returns: attention_weight [batch, num_classes, seq_len], attention_out [batch, num_units]
        """
        # tf.map_fn(lambda x: W_s1 @ x^T, elems=input_x) -> batched W_s1 @ x^T
        h = torch.einsum("ou,bsu->bos", self.W_s1, input_x)  # [batch, attn_unit, seq_len]
        h = torch.tanh(h)
        attention_matrix = torch.einsum(
            "co,bos->bcs", self.W_s2, h
        )  # [batch, num_classes, seq_len]
        attention_weight = F.softmax(attention_matrix, dim=-1)
        attention_out = torch.bmm(attention_weight, input_x)  # [batch, num_classes, num_units]
        attention_out = attention_out.mean(dim=1)  # [batch, num_units]
        return attention_weight, attention_out


class FCLayer(nn.Module):
    def __init__(self, in_dim, fc_hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_dim, fc_hidden_size)
        nn.init.trunc_normal_(self.linear.weight, std=0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, x):
        return F.relu(self.linear(x))


class LocalLayer(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        nn.init.trunc_normal_(self.linear.weight, std=0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, input_x, input_att_weight):
        """
        input_x: [batch, in_dim]
        input_att_weight: [batch, num_classes, seq_len]
        """
        logits = self.linear(input_x)
        scores = torch.sigmoid(logits)
        visual = input_att_weight * scores.unsqueeze(-1)
        visual = F.softmax(visual, dim=-1)
        visual = visual.mean(dim=1)  # [batch, seq_len]
        return logits, scores, visual


class HighwayLayer(nn.Module):
    """Highway Network (Srivastava et al. 2015): t = sigmoid(W'x+b'); h = relu(Wx+b);
    z = t*h + (1-t)*x. Transcribed verbatim from `_highway_layer` (num_layers=1, bias=0)."""

    def __init__(self, size, num_layers=1, bias=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.h_linears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.t_linears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        for lin in self.t_linears:
            nn.init.constant_(lin.bias, bias)

    def forward(self, x):
        for h_lin, t_lin in zip(self.h_linears, self.t_linears):
            h = F.relu(h_lin(x))
            t = torch.sigmoid(t_lin(x))
            x = t * h + (1.0 - t) * x
        return x


class TextHARNN(nn.Module):
    """Faithful port of the real `TextHARNN` TF1 graph (`HARNN/text_harnn.py`)."""

    def __init__(
        self,
        sequence_length,
        vocab_size,
        embedding_size,
        lstm_hidden_size,
        attention_unit_size,
        fc_hidden_size,
        num_classes_list,
        total_classes,
        dropout_rate=0.5,
        alpha=0.5,
    ):
        super().__init__()
        self.num_classes_list = num_classes_list
        self.alpha = alpha

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        self.bilstm = nn.LSTM(
            embedding_size, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True
        )
        self.lstm_dropout = nn.Dropout(1.0 - dropout_rate)

        lstm_out_dim = lstm_hidden_size * 2

        self.attn1 = Attention(lstm_out_dim, attention_unit_size, num_classes_list[0])
        self.fc1 = FCLayer(lstm_out_dim * 2, fc_hidden_size)
        self.local1 = LocalLayer(fc_hidden_size, num_classes_list[0])

        self.attn2 = Attention(lstm_out_dim, attention_unit_size, num_classes_list[1])
        self.fc2 = FCLayer(lstm_out_dim * 2, fc_hidden_size)
        self.local2 = LocalLayer(fc_hidden_size, num_classes_list[1])

        self.attn3 = Attention(lstm_out_dim, attention_unit_size, num_classes_list[2])
        self.fc3 = FCLayer(lstm_out_dim * 2, fc_hidden_size)
        self.local3 = LocalLayer(fc_hidden_size, num_classes_list[2])

        self.attn4 = Attention(lstm_out_dim, attention_unit_size, num_classes_list[3])
        self.fc4 = FCLayer(lstm_out_dim * 2, fc_hidden_size)
        self.local4 = LocalLayer(fc_hidden_size, num_classes_list[3])

        self.fc_final = FCLayer(fc_hidden_size * 4, fc_hidden_size)
        self.highway = HighwayLayer(fc_hidden_size, num_layers=1, bias=0.0)
        self.dropout = nn.Dropout(1.0 - dropout_rate)

        self.global_linear = nn.Linear(fc_hidden_size, total_classes)
        nn.init.trunc_normal_(self.global_linear.weight, std=0.1)
        nn.init.constant_(self.global_linear.bias, 0.1)

    def forward(self, input_x):
        embedded = self.embedding(input_x)  # [batch, seq_len, embedding_size]

        lstm_out, _ = self.bilstm(embedded)  # [batch, seq_len, lstm_hidden*2]
        lstm_out = self.lstm_dropout(lstm_out)
        lstm_out_pool = lstm_out.mean(dim=1)  # [batch, lstm_hidden*2]

        # First level
        first_att_weight, first_att_out = self.attn1(lstm_out)
        first_local_input = torch.cat([lstm_out_pool, first_att_out], dim=1)
        first_fc = self.fc1(first_local_input)
        first_logits, first_scores, first_visual = self.local1(first_fc, first_att_weight)

        # Second level (gated by first_visual)
        second_att_input = lstm_out * first_visual.unsqueeze(-1)
        second_att_weight, second_att_out = self.attn2(second_att_input)
        second_local_input = torch.cat([lstm_out_pool, second_att_out], dim=1)
        second_fc = self.fc2(second_local_input)
        second_logits, second_scores, second_visual = self.local2(second_fc, second_att_weight)

        # Third level (gated by second_visual)
        third_att_input = lstm_out * second_visual.unsqueeze(-1)
        third_att_weight, third_att_out = self.attn3(third_att_input)
        third_local_input = torch.cat([lstm_out_pool, third_att_out], dim=1)
        third_fc = self.fc3(third_local_input)
        third_logits, third_scores, third_visual = self.local3(third_fc, third_att_weight)

        # Fourth level (gated by third_visual)
        fourth_att_input = lstm_out * third_visual.unsqueeze(-1)
        fourth_att_weight, fourth_att_out = self.attn4(fourth_att_input)
        fourth_local_input = torch.cat([lstm_out_pool, fourth_att_out], dim=1)
        fourth_fc = self.fc4(fourth_local_input)
        fourth_logits, fourth_scores, fourth_visual = self.local4(fourth_fc, fourth_att_weight)

        ham_out = torch.cat([first_fc, second_fc, third_fc, fourth_fc], dim=1)
        fc_out = self.fc_final(ham_out)
        highway_out = self.highway(fc_out)
        h_drop = self.dropout(highway_out)

        global_logits = self.global_linear(h_drop)
        global_scores = torch.sigmoid(global_logits)

        local_scores = torch.cat([first_scores, second_scores, third_scores, fourth_scores], dim=1)
        scores = self.alpha * global_scores + (1 - self.alpha) * local_scores

        return scores


def build_harnn():
    return TextHARNN(
        sequence_length=12,
        vocab_size=100,
        embedding_size=16,
        lstm_hidden_size=8,
        attention_unit_size=6,
        fc_hidden_size=10,
        num_classes_list=[3, 4, 5, 6],
        total_classes=18,
        dropout_rate=0.5,
        alpha=0.5,
    )


def example_input_harnn():
    return (torch.randint(0, 100, (2, 12)),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("HARNN", "build_harnn", "example_input_harnn", 2019, "ported"),
]
