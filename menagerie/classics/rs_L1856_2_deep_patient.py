# FAITHFUL PORT of natoromano/deep-patient @ master (original framework: TensorFlow 1.x
# / Python 2, py/auto_encoder.py). The original `StackedEncoders` class builds a greedy
# layer-wise stack of NUM_ENCODERS=3 denoising autoencoders with sigmoid activations and
# H=500 hidden units per layer (per repo defaults): each encoder computes
# `encode = sigmoid(W @ input + b1)`; `decode = sigmoid(W^T @ encode + b2)` (tied weights,
# used only for the greedy per-layer reconstruction loss during training). The final
# "deep patient" representation used downstream is the composed output of stacking all
# `num_encoders` encode operations in sequence (see `_add_encode_ops`: each op reapplies
# `sigmoid(W_i @ x + b1_i)` for i in 0..ops_id). Since the raw code is TF1.x with
# `tf.placeholder`/`tf.InteractiveSession`/Python-2-only `xrange` syntax that cannot run in
# the base torch env, this file transcribes the architecture faithfully into self-contained
# torch: same 3-layer stacked-sigmoid-autoencoder topology, same tied-weight encode/decode
# structure, same hidden_units=500 default. The staged module exposes the inference-time
# "deep patient" encoder stack (the object the paper's downstream classifiers consume),
# matching `_add_encode_ops` with `ops_id = num_encoders - 1`.
"""Faithful torch port of DeepPatient's stacked denoising autoencoder (natoromano/deep-patient)."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DenoisingAutoencoderLayer(nn.Module):
    """One greedy layer of the stack: tied-weight sigmoid encoder/decoder.

    Mirrors `_add_encoder` in the original TF1.x `auto_encoder.py`: a single
    weight matrix `w_` is shared between the encode and decode directions
    (decode uses `w_.T`), matching the original's Xavier-initialized weight
    reused via `tf.transpose(w_)`.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.encode_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decode_bias = nn.Parameter(torch.zeros(input_dim))
        self._xavier_init()

    def _xavier_init(self):
        input_dim, hidden_dim = self.weight.shape
        eps = (6.0 / (input_dim + hidden_dim)) ** 0.5
        with torch.no_grad():
            self.weight.uniform_(-eps, eps)

    def encode(self, x):
        return torch.sigmoid(x @ self.weight + self.encode_bias)

    def decode(self, encoded):
        return torch.sigmoid(encoded @ self.weight.t() + self.decode_bias)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded


class DeepPatientStackedEncoders(nn.Module):
    """The composed "deep patient" representation: NUM_ENCODERS stacked
    denoising-autoencoder encode passes (`_add_encode_ops` with the last,
    fully-stacked op), matching the original repo's NUM_ENCODERS=3, H=500
    defaults.
    """

    def __init__(self, input_dim, hidden_units=500, num_encoders=3):
        super().__init__()
        self.num_encoders = num_encoders
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_encoders):
            self.layers.append(DenoisingAutoencoderLayer(in_dim, hidden_units))
            in_dim = hidden_units

    def forward(self, x):
        for layer in self.layers:
            x = layer.encode(x)
        return x


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny sizes, matches the repo's own
# stacked-DAE topology, scaled down for fast tracing).
# ---------------------------------------------------------------------------


def build_deep_patient():
    return DeepPatientStackedEncoders(input_dim=64, hidden_units=32, num_encoders=3)


def example_input_deep_patient():
    return (torch.rand(4, 64),)


MENAGERIE_ENTRIES = [
    ("DeepPatient", "build_deep_patient", "example_input_deep_patient", 2016, "ported-pytorch"),
]
