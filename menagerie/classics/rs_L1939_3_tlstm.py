# FAITHFUL PORT of illidanlab/T-LSTM @ master (original framework: TensorFlow 1.x, Python 2.7)
# https://raw.githubusercontent.com/illidanlab/T-LSTM/master/TLSTM.py
#
# "Patient Subtyping via Time-Aware LSTM Networks" (Baytas, Xiao, Zhang, Wang, Jain, Zhou;
# KDD 2017). T-LSTM = a single-layer LSTM cell modified with an explicit elapsed-time decay: at
# each step the previous cell state is decomposed into a short-term component that decays with
# elapsed time `t` (`map_elapse_time`) and a long-term component that is carried over
# unchanged, before running the usual input/forget/output/candidate gates. This is a faithful
# transcription of the real `TLSTM` class's `TLSTM_Unit` / `get_states` / `get_output` /
# `get_outputs` methods (TF1.x `tf.placeholder` / `tf.get_variable` / `tf.scan` API, unrunnable
# unmodified since TF2.x removed the v1 graph-mode placeholder/variable_scope surface used
# throughout) into a self-contained `torch.nn.Module`:
#   - `init_weights`/`init_bias`/`no_init_weights`/`no_init_bias` (train-vs-eval variable
#     creation toggle, `train` ctor flag) collapse into ordinary `nn.Parameter`s -- torch has no
#     analogous train/eval variable-creation branch, and the two branches allocate identical
#     shapes.
#   - `TLSTM_Unit(prev_hidden_memory, concat_input)`: cell-state time-decomposition
#     (`C_ST = tanh(prev_cell @ W_decomp + b_decomp)`, `T = map_elapse_time(t)`,
#     `prev_cell = prev_cell - C_ST + T * C_ST`) followed by standard LSTM gates
#     (input/forget/output/candidate, each `sigmoid`/`sigmoid`/`sigmoid`/`tanh` of
#     `x @ W + h_prev @ U + b`) -- transcribed verbatim as `_t_lstm_unit`.
#   - `map_elapse_time(t)`: `T = 1 / log(t + e)` broadcast across the hidden dim -- transcribed
#     verbatim as `_map_elapse_time` (the real code's literal `2.7183` constant for Euler's
#     number is kept rather than `math.e`, matching the source exactly).
#   - `get_states()` (`tf.scan` over the time axis) -> ordinary Python `for` loop over the
#     sequence dimension, identical recurrence.
#   - `get_output`/`get_outputs` (FC + dropout + softmax-logit head applied to *every* time
#     step via `tf.map_fn`, then only the *last* (`tf.reverse(...)[0]`) time step's output is
#     kept) -- transcribed as `get_outputs()` returning just the final-step logits, matching
#     what the real `get_cost_acc()` consumes.
#   - `get_cost_acc` (cross-entropy training loss) is training infrastructure, not part of the
#     forward architecture, and is dropped; the module's `forward()` returns the final-step
#     logits that `get_cost_acc` would have consumed.
import math

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class TLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        def w(in_d, out_d):
            return nn.Parameter(torch.empty(in_d, out_d).normal_(0.0, 0.1))

        def b(out_d):
            return nn.Parameter(torch.ones(out_d))

        # input gate
        self.Wi = w(input_dim, hidden_dim)
        self.Ui = w(hidden_dim, hidden_dim)
        self.bi = b(hidden_dim)

        # forget gate
        self.Wf = w(input_dim, hidden_dim)
        self.Uf = w(hidden_dim, hidden_dim)
        self.bf = b(hidden_dim)

        # output gate
        self.Wog = w(input_dim, hidden_dim)
        self.Uog = w(hidden_dim, hidden_dim)
        self.bog = b(hidden_dim)

        # candidate memory cell
        self.Wc = w(input_dim, hidden_dim)
        self.Uc = w(hidden_dim, hidden_dim)
        self.bc = b(hidden_dim)

        # elapsed-time cell decomposition
        self.W_decomp = w(hidden_dim, hidden_dim)
        self.b_decomp = b(hidden_dim)

        # FC + output head
        self.Wo = w(hidden_dim, fc_dim)
        self.bo = b(fc_dim)
        self.W_softmax = w(fc_dim, output_dim)
        self.b_softmax = b(output_dim)

    def _map_elapse_time(self, t):
        # T = 1 / log(t + e); broadcast to hidden_dim. `t`: (batch, 1)
        c1 = 1.0
        c2 = 2.7183
        T = c1 / torch.log(t + c2)
        ones = torch.ones(1, self.hidden_dim, dtype=T.dtype, device=T.device)
        T = T.matmul(ones)  # (batch, hidden_dim)
        return T

    def _t_lstm_unit(self, prev_hidden_state, prev_cell, x, t):
        T = self._map_elapse_time(t)

        c_st = torch.tanh(prev_cell.matmul(self.W_decomp) + self.b_decomp)
        c_st_dis = T * c_st
        prev_cell = prev_cell - c_st + c_st_dis

        i = torch.sigmoid(x.matmul(self.Wi) + prev_hidden_state.matmul(self.Ui) + self.bi)
        f = torch.sigmoid(x.matmul(self.Wf) + prev_hidden_state.matmul(self.Uf) + self.bf)
        o = torch.sigmoid(x.matmul(self.Wog) + prev_hidden_state.matmul(self.Uog) + self.bog)
        c_tilde = torch.tanh(x.matmul(self.Wc) + prev_hidden_state.matmul(self.Uc) + self.bc)

        c = f * prev_cell + i * c_tilde
        h = o * torch.tanh(c)
        return h, c

    def get_states(self, x, t):
        # x: (batch, seq_len, input_dim), t: (batch, seq_len)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        all_states = []
        for step in range(seq_len):
            x_t = x[:, step, :]
            t_t = t[:, step : step + 1]
            h, c = self._t_lstm_unit(h, c, x_t, t_t)
            all_states.append(h)
        return torch.stack(all_states, dim=1)  # (batch, seq_len, hidden_dim)

    def get_output(self, state):
        output = torch.relu(state.matmul(self.Wo) + self.bo)
        output = self.dropout(output)
        output = output.matmul(self.W_softmax) + self.b_softmax
        return output

    def get_outputs(self, x, t):
        all_states = self.get_states(x, t)
        # real repo applies get_output per time step then keeps only the last step
        # (`tf.reverse(all_outputs, [0])[0, :, :]`) -- equivalent to just the final state.
        return self.get_output(all_states[:, -1, :])

    def forward(self, x, t):
        return self.get_outputs(x, t)


def build_tlstm():
    torch.manual_seed(0)
    model = TLSTM(input_dim=8, output_dim=2, hidden_dim=16, fc_dim=12, dropout=0.0)
    model.eval()
    return model


def example_input_tlstm():
    torch.manual_seed(0)
    batch_size = 4
    seq_len = 6
    input_dim = 8
    x = torch.randn(batch_size, seq_len, input_dim)
    # elapsed time between visits (strictly positive; map_elapse_time takes log(t + e))
    t = torch.rand(batch_size, seq_len) * 10.0 + 0.1
    return (x, t)


MENAGERIE_ENTRIES = [
    ("T-LSTM", "build_tlstm", "example_input_tlstm", 2017, "ported"),
]
