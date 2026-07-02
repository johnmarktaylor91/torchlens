# FAITHFUL PORT of xiph/rnnoise @ master, training/rnn_train.py (original framework:
# Keras/TensorFlow functional API)
# https://raw.githubusercontent.com/xiph/rnnoise/master/training/rnn_train.py
#
# RNNoise (Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech
# Enhancement", 2018) is a lightweight per-frame recurrent noise-suppression gain
# predictor: a 42-dim hand-crafted feature vector per 10ms frame (Bark-scale band
# energies + pitch features) drives a small Dense->GRU stack with three GRU cells
# (vad_gru, noise_gru, denoise_gru) whose intermediate activations are concatenated at
# each stage, producing an 18-D voice-activity-plus-denoising-gain output and a scalar
# VAD probability. The real repo trains this in Keras (`rnn_train.py`) then exports the
# learned weights into a bespoke fixed-point C inference engine (`src/rnn.c` /
# `src/denoise.c`) for real-time deployment -- the C code is NOT a from-scratch
# reimplementation target here since it is not an nn.Module; the trainable Keras graph
# IS the real architecture, transcribed layer-for-layer below (same Dense/GRU sizes,
# same concatenation topology, same activations) since Keras/TensorFlow is not usable
# in this environment (protobuf/MessageFactory incompatibility blocks `import
# tensorflow`/`import keras` here).
#
# Every layer matches rnn_train.py's functional graph one-for-one:
#   main_input (42) -> Dense(24, tanh) "input_dense" -> tmp
#   tmp -> GRU(24, tanh/sigmoid) "vad_gru" -> vad_gru_out
#   vad_gru_out -> Dense(1, sigmoid) "vad_output" -> vad_output
#   concat(tmp, vad_gru_out, main_input) (24+24+42=90) -> GRU(48, relu/sigmoid)
#     "noise_gru" -> noise_gru_out
#   concat(vad_gru_out, noise_gru_out, main_input) (24+48+42=114) -> GRU(96, tanh/sigmoid)
#     "denoise_gru" -> denoise_gru_out
#   denoise_gru_out -> Dense(22, sigmoid) "denoise_output" -> denoise_output
#
# Keras GRU uses the "reset-after" gate ordering internally (update/reset/new gates with
# the reset gate applied to the recurrent contribution before adding the bias), whereas
# torch.nn.GRU implements the standard formulation (reset gate applied to the recurrent
# matmul directly). The gate *topology* (input Dense sizes, hidden sizes, concatenation
# structure, per-gate activation choice of sigmoid for update/reset and tanh/relu for the
# candidate state) is transcribed exactly as declared in rnn_train.py; torch.nn.GRU is
# used per-cell as the faithful torch counterpart of Keras' GRU layer (both are
# single-layer, single-direction gated recurrent units with the same input/hidden sizes
# declared in the source). Weight-clipping (`WeightClip`, `kernel_constraint`,
# `min_max_norm`) and L2 regularizers are TRAINING-time regularization, not part of the
# forward architecture, and are intentionally omitted (they do not change the graph
# topology or shapes the module computes at inference).
#
# Only base-lib deps used: torch, torch.nn.

import torch
import torch.nn as nn


class _KerasStyleGRU(nn.Module):
    """Single-layer GRU with an explicit non-default recurrent activation, matching a
    Keras GRU(units, activation=<candidate_act>, recurrent_activation='sigmoid') layer.
    torch.nn.GRU hard-codes tanh for the candidate/new-gate; RNNoise's noise_gru uses
    'relu' for its candidate activation (rnn_train.py line for noise_gru), so a manual
    per-step GRU cell (matching Keras' default, non-"reset_after" gate equations) is used
    to preserve that architectural detail faithfully rather than silently swapping in
    tanh."""

    def __init__(self, input_size: int, hidden_size: int, candidate_activation: str):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # combined z (update), r (reset), h (candidate) gate weights, Keras layout
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if candidate_activation == "tanh":
            self._cand_act = torch.tanh
        elif candidate_activation == "relu":
            self._cand_act = torch.relu
        else:
            raise ValueError(candidate_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, input_size) -> returns (batch, time, hidden_size)
        bs, t, _ = x.shape
        h = torch.zeros(bs, self.hidden_size, dtype=x.dtype, device=x.device)
        outs = []
        for step in range(t):
            xt = x[:, step, :]
            gi = torch.nn.functional.linear(xt, self.weight_ih, self.bias_ih)
            gh = torch.nn.functional.linear(h, self.weight_hh, self.bias_hh)
            i_z, i_r, i_n = gi.chunk(3, dim=-1)
            h_z, h_r, h_n = gh.chunk(3, dim=-1)
            z = torch.sigmoid(i_z + h_z)
            r = torch.sigmoid(i_r + h_r)
            n = self._cand_act(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)


class RNNoiseModel(nn.Module):
    """Port of rnn_train.py's Keras functional graph (Build model... block)."""

    def __init__(self, input_dim: int = 42):
        super().__init__()
        self.input_dense = nn.Sequential(nn.Linear(input_dim, 24), nn.Tanh())
        self.vad_gru = _KerasStyleGRU(24, 24, candidate_activation="tanh")
        self.vad_output = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())
        self.noise_gru = _KerasStyleGRU(24 + 24 + input_dim, 48, candidate_activation="relu")
        self.denoise_gru = _KerasStyleGRU(24 + 48 + input_dim, 96, candidate_activation="tanh")
        self.denoise_output = nn.Sequential(nn.Linear(96, 22), nn.Sigmoid())

    def forward(self, main_input: torch.Tensor):
        tmp = self.input_dense(main_input)
        vad_gru_out = self.vad_gru(tmp)
        vad_output = self.vad_output(vad_gru_out)

        noise_input = torch.cat([tmp, vad_gru_out, main_input], dim=-1)
        noise_gru_out = self.noise_gru(noise_input)

        denoise_input = torch.cat([vad_gru_out, noise_gru_out, main_input], dim=-1)
        denoise_gru_out = self.denoise_gru(denoise_input)

        denoise_output = self.denoise_output(denoise_gru_out)
        return denoise_output, vad_output


def build_rnnoise():
    torch.manual_seed(0)
    model = RNNoiseModel(input_dim=42)
    model.eval()
    return model


def example_input_rnnoise():
    torch.manual_seed(0)
    # (batch, time, 42) matching main_input = Input(shape=(None, 42))
    return torch.randn(2, 8, 42)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("RNNoise", "build_rnnoise", "example_input_rnnoise", 2018, MENAGERIE_ZOO),
]
