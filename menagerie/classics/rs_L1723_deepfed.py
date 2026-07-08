# FAITHFUL PORT of https://github.com/cs-joy/DeepFED_EXPLO @ main (Model.py,
# model_train.py) (original framework: TensorFlow/Keras)
#
# DeepFed: federated CNN-GRU intrusion detector for industrial cyber-physical
# systems (Li et al., IEEE Trans. Industrial Informatics 2020, "DeepFed:
# Federated Deep Learning for Intrusion Detection in Industrial Cyber-Physical
# Systems"). No official author repo was found (GitHub/PyPI search turned up
# only a Paillier-encryption federated-aggregation reimplementation with no
# accompanying model code, plus unrelated GRU variants from other repos'
# own experiments); `cs-joy/DeepFED_EXPLO` ("An implementation of DeepFED")
# is the one community repo whose `Model.create_cnn_gru_model` matches the
# architecture described in the queue (CNN-GRU IDS) and the paper: it is
# real, runnable TF/Keras code with a documented example usage
# (input_shape=(50, 1), num_classes=10 in model_train.py).
#
# This is a straight per-layer transcription into torch of that Keras
# `Sequential`:
#   Conv1D(filters=64, kernel_size=3, activation='relu')
#   -> MaxPooling1D(pool_size=2)
#   -> GRU(units=128, return_sequences=False)
#   -> Dense(num_classes, activation='softmax')
# Keras Conv1D/GRU use channel-last (batch, steps, channels); torch's
# nn.Conv1d/nn.GRU(batch_first=True) need explicit permutes around the conv
# to match. GRU(return_sequences=False) in Keras returns only the final
# timestep's hidden state, mirrored here by indexing `out[:, -1, :]`. No
# layer, filter count, kernel size, or ordering was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepFedCNNGRU(nn.Module):
    """CNN-GRU federated intrusion-detection client model: one Conv1D+pool
    feature-extraction block feeding a single-layer GRU classifier head."""

    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        conv_filters=64,
        kernel_size=3,
        gru_units=128,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, conv_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(conv_filters, gru_units, batch_first=True)
        self.fc = nn.Linear(gru_units, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, in_channels) -- Keras Conv1D channel-last input.
        x = x.permute(0, 2, 1)  # -> (batch, in_channels, seq_len) for nn.Conv1d
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)  # -> (batch, seq', conv_filters) for GRU(batch_first=True)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # Keras GRU(return_sequences=False): final timestep only
        logits = self.fc(last)
        return torch.softmax(logits, dim=-1)


def build_deepfed():
    # Matches model_train.py's documented example usage exactly:
    # input_shape=(50, 1), num_classes=10.
    return DeepFedCNNGRU(in_channels=1, num_classes=10)


def example_input_deepfed():
    # (batch, seq_len=50, in_channels=1), the real repo's example time-series window.
    return torch.randn(4, 50, 1)


MENAGERIE_ENTRIES = [
    (
        "DeepFed",
        build_deepfed,
        example_input_deepfed,
        2020,
        MENAGERIE_ZOO,
    ),
]
