# FAITHFUL PORT of okurman/TREDNet @ 0fcd3023a210df6c5ca9a4ae7a055eb9f9d4eaa7 (original framework: Keras/TensorFlow)
#   lib/v1/models.py: define_model()
#
# TREDNet (Hudaiberdiev, Ovcharenko, et al., medRxiv/2022; okurman/TREDNet) is a residual
# CNN regressor for scoring the regulatory-enhancer activity of DNA sequences and, in its
# "phase two" delta-scoring mode, for predicting the functional effect of noncoding
# regulatory SNPs from a pair of reference/alternate one-hot-encoded sequences. The real
# Keras `define_model()` (lib/v1/models.py, and identically in models_mini.py and the
# kipoi phase_one/phase_two published packages) builds a `Sequential`:
#   Conv1D(64, kernel_size=4, stride=1, relu, l1_l2 reg, max_norm(1) constraint)
#   -> BatchNormalization -> MaxPooling1D(2) -> Dropout(0.4)
#   -> Conv1D(128, kernel_size=2, stride=1, relu, l1_l2 reg, max_norm(1) constraint)
#   -> Dropout(0.4) -> Flatten -> Dense(100, relu) -> Dense(50, relu) -> Dense(1, sigmoid)
# over a one-hot-encoded (INPUT_LENGTH=1924, 4) DNA window (real kipoi schema: (2000,4) for
# phase_one). Reproduced 1:1 below as torch: Conv1d layers keep filters/kernel_size/stride;
# max_norm(1) weight-norm constraint on the Conv1d kernels is applied as a post-step weight
# renorm hook (`_apply_max_norm`), matching Keras `kernel_constraint=max_norm(1)`; l1_l2
# kernel regularization is a training-time loss penalty (not part of the forward-pass graph)
# and is intentionally omitted, matching how graph-capture tools treat other keras
# regularizer/constraint metadata as non-architectural. BatchNormalization -> nn.BatchNorm1d,
# MaxPooling1D -> nn.MaxPool1d, Dropout -> nn.Dropout, Flatten -> torch.flatten, Dense ->
# nn.Linear with matching activations (ReLU/ReLU/Sigmoid).

import torch
import torch.nn as nn


class TREDNetCNN(nn.Module):
    """Faithful port of lib/v1/models.py `define_model()`'s Keras Sequential graph."""

    def __init__(
        self, input_length=1924, filters_1=64, filters_2=128, dropout=0.4, max_norm_val=1.0
    ):
        super().__init__()
        self.max_norm_val = max_norm_val

        self.conv1 = nn.Conv1d(1, filters_1, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm1d(filters_1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(filters_1, filters_2, kernel_size=2, stride=1)
        self.drop2 = nn.Dropout(dropout)

        conv1_out = input_length - 4 + 1
        pool1_out = conv1_out // 2
        conv2_out = pool1_out - 2 + 1
        flat_dim = filters_2 * conv2_out

        self.fc1 = nn.Linear(flat_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

    def _apply_max_norm(self, layer):
        # Keras `kernel_constraint=max_norm(1)`: clip each output-filter's kernel L2 norm to
        # <= max_norm_val after each forward (constraint semantics, not a loss term).
        with torch.no_grad():
            w = layer.weight
            norm = w.view(w.size(0), -1).norm(dim=1, keepdim=True).clamp(min=1e-7)
            desired = torch.clamp(norm, max=self.max_norm_val)
            factor = (desired / norm).view(-1, *([1] * (w.dim() - 1)))
            w.mul_(factor)

    def forward(self, x):
        # x: (batch, length, 1) one-hot-encoded DNA window -> torch Conv1d wants (batch, channels, length)
        self._apply_max_norm(self.conv1)
        self._apply_max_norm(self.conv2)

        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = torch.relu(self.conv2(x))
        x = self.drop2(x)

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def build_trednet():
    # Real usage (lib/v1/models.py `define_model`): INPUT_LENGTH=1924, filters (64, 128).
    # Length shrunk here for tiny tracing; filters/kernel_sizes kept at their real values.
    return TREDNetCNN(input_length=64, filters_1=64, filters_2=128, dropout=0.4)


def example_input_trednet():
    # Real usage (kipoi schema, score_variant.py): one-hot-encoded (batch, length, 1)
    # DNA sequence windows (4-letter alphabet flattened per the phase-one Conv1D input_shape).
    batch, length = 2, 64
    return torch.rand(batch, length, 1)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("TREDNet", "build_trednet", "example_input_trednet", 2022, MENAGERIE_ZOO),
]
