# FAITHFUL PORT of mp2893/doctorai @ master (original framework: Theano, Python 2)
#
# Source: doctorAI.py -- Edward Choi's "Doctor AI: Predicting Clinical Events via
# Recurrent Neural Networks" (MLHC 2016). init_params/gru_layer/build_model define the
# architecture: a tanh embedding of multi-hot medical-code vectors (optionally
# concatenated with an elapsed-time scalar), a STACK of hand-rolled GRU layers (the
# repo implements the GRU gate equations manually with explicit W/U/b parameter
# matrices rather than using a framework GRU cell -- transcribed 1:1 below, same
# r/z/h_tilde equations, same per-layer dropout), and two output heads read from the
# final hidden state at every timestep: a softmax code-prediction head (`W_output`/
# `b_output`) and an optional ReLU visit-duration regression head (`W_time`/`b_time`,
# active when `predictTime`).
#
# The original repo is Python 2 + Theano (`theano.scan`, `cPickle`, `iteritems`) and
# cannot run in a torch base env (Theano is unmaintained/EOL, incompatible with modern
# Python). This port transcribes the SAME per-timestep GRU equations
# (init_params/gru_layer) and the SAME embedding -> stacked-GRU -> dual-head structure
# (build_model) into eager torch, replacing `theano.scan` with an explicit Python time
# loop over the (batch-mask-respecting) input sequence -- semantically identical to the
# original scan for a fixed, statically-known sequence length. Training-only pieces
# (adadelta optimizer, dropout-noise RNG plumbing, embedding-file loading) are not part
# of the architecture and are not ported; dropout is present as a real nn.Dropout layer
# matching `dropout_layer`, active only in training mode as in the original.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DoctorAIGRULayer(nn.Module):
    """One hand-rolled GRU layer, per gru_layer() in the original repo: explicit
    reset/update/candidate gates with separate input (W) and recurrent (U) weight
    matrices, masked per-timestep so padded steps carry the previous hidden state
    forward unchanged."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_r = nn.Parameter(torch.empty(input_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.W_z = nn.Parameter(torch.empty(input_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.W = nn.Parameter(torch.empty(input_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.U_r = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.U_z = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.U = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-0.01, 0.01))
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, emb, mask):
        # emb: (T, B, input_dim), mask: (T, B)
        T, B, _ = emb.shape
        w_rx = emb @ self.W_r
        w_zx = emb @ self.W_z
        wx = emb @ self.W

        h = torch.zeros(B, self.hidden_dim, dtype=emb.dtype, device=emb.device)
        outs = []
        for t in range(T):
            r = torch.sigmoid(w_rx[t] + h @ self.U_r + self.b_r)
            z = torch.sigmoid(w_zx[t] + h @ self.U_z + self.b_z)
            h_tilde = torch.tanh(wx[t] + (r * h) @ self.U + self.b)
            h_new = z * h + (1.0 - z) * h_tilde
            step_mask = mask[t].unsqueeze(-1)
            h = step_mask * h_new + (1.0 - step_mask) * h
            outs.append(h)
        return torch.stack(outs, dim=0)


class DoctorAI(nn.Module):
    """Stacked-GRU EHR event/code predictor, per build_model() in the original repo
    (embFineTune=True path: the code embedding W_emb is a learned parameter, not a
    frozen pretrained lookup)."""

    def __init__(
        self,
        input_dim_size=40,
        num_class=30,
        emb_size=32,
        hidden_dim_sizes=(48, 48),
        dropout_rate=0.5,
        use_time=True,
        predict_time=True,
    ):
        super().__init__()
        self.use_time = use_time
        self.predict_time = predict_time

        self.W_emb = nn.Parameter(torch.empty(input_dim_size, emb_size).uniform_(-0.01, 0.01))
        self.b_emb = nn.Parameter(torch.zeros(emb_size))

        prev_dim = emb_size + (1 if use_time else 0)
        layers = []
        for hidden_dim in hidden_dim_sizes:
            layers.append(DoctorAIGRULayer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.gru_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_rate)

        self.W_output = nn.Parameter(torch.empty(prev_dim, num_class).uniform_(-0.01, 0.01))
        self.b_output = nn.Parameter(torch.zeros(num_class))

        if predict_time:
            self.W_time = nn.Parameter(torch.empty(prev_dim, 1).uniform_(-0.01, 0.01))
            self.b_time = nn.Parameter(torch.zeros(1))

    def forward(self, x, t, mask):
        """x: (T, B, input_dim_size) multi-hot code vectors, t: (T, B) elapsed-time
        scalar (used only if use_time), mask: (T, B) 1.0 for real timesteps."""
        emb = torch.tanh(x @ self.W_emb + self.b_emb)
        if self.use_time:
            emb = torch.cat([t.unsqueeze(-1), emb], dim=2)

        h = emb
        for layer in self.gru_layers:
            h = layer(h, mask)
            h = self.dropout(h)

        T, B, _ = h.shape
        logits = h.reshape(T * B, -1) @ self.W_output + self.b_output
        probs = torch.softmax(logits, dim=-1).view(T, B, -1)
        probs = probs * mask.unsqueeze(-1)

        if self.predict_time:
            duration = torch.relu(h @ self.W_time + self.b_time).squeeze(-1) * mask
            return probs, duration
        return probs


def build_doctorai():
    torch.manual_seed(0)
    return DoctorAI(
        input_dim_size=40,
        num_class=30,
        emb_size=32,
        hidden_dim_sizes=(48, 48),
        dropout_rate=0.5,
        use_time=True,
        predict_time=True,
    ).eval()


def example_input_doctorai():
    torch.manual_seed(0)
    T, B, input_dim = 6, 3, 40
    x = torch.zeros(T, B, input_dim)
    for t in range(T):
        for b in range(B):
            idx = torch.randint(0, input_dim, (3,))
            x[t, b, idx] = 1.0
    t = torch.rand(T, B)
    mask = torch.ones(T, B)
    return (x, t, mask)


MENAGERIE_ENTRIES = [
    ("DoctorAI", build_doctorai, example_input_doctorai, 2016, "ported-pytorch"),
]
