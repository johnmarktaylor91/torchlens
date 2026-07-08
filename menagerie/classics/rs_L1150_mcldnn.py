# FAITHFUL PORT of Xiaohan-Chen/bear_fault_diagnosis @ master (original framework: Keras/TensorFlow)
# https://github.com/Xiaohan-Chen/bear_fault_diagnosis/blob/master/cnn_lstm_model.ipynb
#
# Official baseline for: Chen, X., Zhang, B. & Gao, D. "Bearing fault diagnosis base on multi-scale
# CNN and LSTM model." J Intell Manuf (2020). https://doi.org/10.1007/s10845-020-01600-2 (MCLDNN:
# Multi-scale CNN-LSTM DNN for bearing fault diagnosis).
#
# The original `built_model()` (Keras, in `cnn_lstm_model.ipynb`) is a dual-encoder architecture:
#   - encoder1: 2x Conv1D (k=20/s=2, k=10/s=2) + MaxPool1D(2)               -> (channels_first)
#   - encoder2: 4x Conv1D (k=6/s=1, k=6/s=1, [pool], k=6/s=1, k=6/s=2) + 2x MaxPool1D(2)
#   - the two encoder outputs are fused via elementwise `multiply`
#   - 2x LSTM (60 units, first `return_sequences=True`) -> Dropout(0.5) -> Dense(10, softmax)
#
# This is transcribed layer-for-layer into self-contained torch (Keras is not an installed base
# lib here, so the real Keras code cannot be run/vendored directly per rung 2; this is a faithful
# rung-3 port of the exact same ops/shapes/hyperparameters).

import torch
import torch.nn as nn


class MCLDNN(nn.Module):
    """Multi-scale CNN-LSTM DNN for bearing fault diagnosis (Chen, Zhang & Gao, 2020)."""

    def __init__(self, in_length: int = 250, n_classes: int = 10):
        super().__init__()
        self.in_length = in_length

        # Encoder 1: two-stage multi-scale conv branch.
        self.enc1_conv1 = nn.Conv1d(1, 50, kernel_size=20, stride=2)
        self.enc1_conv2 = nn.Conv1d(50, 30, kernel_size=10, stride=2)
        self.enc1_pool = nn.MaxPool1d(kernel_size=2)

        # Encoder 2: deeper multi-scale conv branch.
        self.enc2_conv1 = nn.Conv1d(1, 50, kernel_size=6, stride=1)
        self.enc2_conv2 = nn.Conv1d(50, 40, kernel_size=6, stride=1)
        self.enc2_pool1 = nn.MaxPool1d(kernel_size=2)
        self.enc2_conv3 = nn.Conv1d(40, 30, kernel_size=6, stride=1)
        self.enc2_conv4 = nn.Conv1d(30, 30, kernel_size=6, stride=2)
        self.enc2_pool2 = nn.MaxPool1d(kernel_size=2)

        self.tanh = nn.Tanh()

        # Decoder: fused features -> 2-layer LSTM -> dropout -> softmax classifier.
        self.lstm1 = nn.LSTM(input_size=30, hidden_size=60, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=60, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(60, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_length) raw 1D vibration signal window.
        Returns:
            (batch, n_classes) class probabilities.
        """
        x = x.unsqueeze(1)  # (batch, 1, in_length), channels-first like the Keras Reshape((1, L)).

        # Encoder 1.
        e1 = self.tanh(self.enc1_conv1(x))
        e1 = self.tanh(self.enc1_conv2(e1))
        e1 = self.enc1_pool(e1)

        # Encoder 2.
        e2 = self.tanh(self.enc2_conv1(x))
        e2 = self.tanh(self.enc2_conv2(e2))
        e2 = self.enc2_pool1(e2)
        e2 = self.tanh(self.enc2_conv3(e2))
        e2 = self.tanh(self.enc2_conv4(e2))
        e2 = self.enc2_pool2(e2)

        fused = e1 * e2  # elementwise `multiply`, both (batch, 30, 27).
        fused = fused.transpose(1, 2)  # (batch, seq_len=27, channels=30) for batch_first LSTM.

        out, _ = self.lstm1(fused)
        out, (h_n, _) = self.lstm2(out)
        out = h_n[-1]  # final hidden state, equivalent to Keras LSTM(60) without return_sequences.

        out = self.dropout(out)
        out = self.classifier(out)
        return self.softmax(out)


# ==================================================================================================
# MENAGERIE staging entry points

MENAGERIE_ZOO = "ported-pytorch"


def build_mcldnn():
    return MCLDNN(in_length=250, n_classes=10)


def example_input_mcldnn():
    return torch.randn(2, 250)


MENAGERIE_ENTRIES = [
    ("MCLDNN", build_mcldnn, example_input_mcldnn, 2020, "ported-pytorch"),
]
