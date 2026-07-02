# SOURCE: vendored from passalis/dain @ master (dain.py)
# https://raw.githubusercontent.com/passalis/dain/master/dain.py
#
# Deep Adaptive Input Normalization (DAIN) for Time Series Forecasting
# (Passalis et al., IEEE TNNLS 2020). A single learnable normalization layer with
# three cascaded, parametrized steps -- adaptive shift (linear map over the
# per-channel mean), adaptive scale (linear map over the per-channel std), and an
# adaptive gate (sigmoid-gated rescaling) -- applied to the (n_samples, dim,
# n_feature_vectors) input before it is consumed by a downstream forecasting
# model. `mode='full'` (the paper's complete 3-step variant) is used here.
#
# Minimal API-compat fixes (NOT architecture changes):
#   - `tensor.resize(...)` (deprecated/unsafe in-place reshape removed from modern
#     torch semantics) -> `tensor.reshape(...)` (identical (n, d, 1) reshape here).
#   - `F.sigmoid(x)` (deprecated functional alias) -> `torch.sigmoid(x)`.
# Everything else (the three normalization steps, learnable identity-initialized
# linear layers, epsilon handling) is untouched from the original source.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAIN_Layer(nn.Module):
    def __init__(
        self, mode="adaptive_avg", mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144
    ):
        super(DAIN_Layer, self).__init__()

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode is None:
            pass

        # Do simple average normalization
        elif self.mode == "avg":
            avg = torch.mean(x, 2)
            avg = avg.reshape(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == "adaptive_avg":
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == "adaptive_scale":
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x**2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == "full":
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x**2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, 2)
            gate = torch.sigmoid(self.gating_layer(avg))
            gate = gate.reshape(gate.size(0), gate.size(1), 1)
            x = x * gate

        else:
            assert False

        return x


MENAGERIE_ZOO = "vendored-pytorch"


def build_dain():
    model = DAIN_Layer(mode="full", input_dim=16)
    model.eval()
    return model


def example_input_dain():
    # (n_samples, dim, n_feature_vectors)
    return torch.randn(4, 16, 20)


MENAGERIE_ENTRIES = [
    (
        "DAIN (Deep Adaptive Input Normalization)",
        "build_dain",
        "example_input_dain",
        2020,
        MENAGERIE_ZOO,
    ),
]
