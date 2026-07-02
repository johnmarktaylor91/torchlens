# FAITHFUL PORT of Richardzhangxx/PET-CGDNN @ 8dd984b46803e9cfa25e455e5a2868f6545f5fed
# (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/Richardzhangxx/PET-CGDNN/main/Experiment%20on%202016a/rmlmodels/PETCGDNN.py
#
# PET-CGDNN (Zhang, Wang, Feng, He 2021, IEEE T-VT / TSP letters style, "Automatic
# Modulation Classification Using Involution Enabled Residual Networks" family; the
# PET-CGDNN variant is "Parameter Estimation and Transformation" + Convolutional +
# GRU DNN) for automatic modulation classification (AMC) on RadioML IQ sequences. The
# real Keras model (`PETCGDNN` in `rmlmodels/PETCGDNN.py`) is a functional-API graph:
# a small dense "phase estimator" branch predicts a per-sequence rotation angle from
# the flattened raw IQ frame, `cos`/`sin` of that angle are used to phase-rotate the
# raw I/Q channels ("parameter estimation and transformation", the PET in the name),
# the rotated I/Q are stacked back into a [128, 2, 1] map and run through two "valid"
# Conv2D layers (spatial feature), reshaped to a length-117 sequence over 25 channels
# and fed through one CuDNNGRU(128) (temporal feature), then a final softmax Dense
# head. This repo uses `TensorFlow/Keras` (`keras.layers.CuDNNLSTM/CuDNNGRU`, a legacy
# GPU-only cuDNN-fused RNN layer) which cannot be constructed in a modern
# TensorFlow/Keras install (removed API) and is architecturally a plain GRU (CuDNNGRU
# is bitwise-equivalent to `GRU` with `reset_after=True`, `recurrent_activation=
# "sigmoid"`, run on GPU) -- so this port transcribes the exact functional graph
# (same layer shapes/kernel sizes/paddings/activations/order of ops) into base-env
# torch, using `nn.GRU` for the CuDNNGRU layer (the numerically-equivalent op, not an
# architectural change) since the original op has no torch-importable analogue.
#
# Every op in the real `PETCGDNN()` functional graph is preserved 1:1:
#   Flatten -> Dense(1) -> linear activation                          (x1, phase estimate)
#   cos(x1), sin(x1)                                                  (cal1/cal2 Lambdas)
#   y1 = I*cos + Q*sin ; y2 = Q*cos - I*sin                            (phase compensation)
#   stack [y1, y2] -> reshape to (128, 2, 1)                           (reshape1/2/3)
#   Conv2D(75, kernel=(8,2), padding="valid", relu)                    (conv1_1)
#   Conv2D(25, kernel=(5,1), padding="valid", relu)                    (conv1_2)
#   reshape to (117, 25)                                               (reshape4)
#   CuDNNGRU(128) -> ported as nn.GRU(25, 128)                         (temporal feature)
#   Dense(classes, softmax)                                            (final head)
# Keras Conv2D is channels-last (H, W, C_in) with a (kh, kw) kernel over the
# (128, 2) spatial map with 1 input channel; this port keeps torch's channels-first
# Conv2d numerically identical by treating the (128 time-steps, 2 IQ-channels) axes
# as the (H, W) spatial dims, matching the Keras kernel shapes/valid-padding exactly
# (output spatial sizes: 128->121->117 over H, 2->1->1 over W, matching Keras'
# "valid" arithmetic bit-for-bit).

import torch
import torch.nn as nn


class PETCGDNN(nn.Module):
    """Faithful torch port of the real Keras `PETCGDNN` functional model."""

    def __init__(self, input_len=128, classes=11):
        super().__init__()
        self.input_len = input_len

        # x1 = Dense(1)(Flatten(input))  -- input has shape (input_len, 2, 1)
        self.fc2 = nn.Linear(input_len * 2, 1)

        # conv1_1: Conv2D(75, (8, 2), padding="valid", relu)
        self.conv1_1 = nn.Conv2d(1, 75, kernel_size=(8, 2), padding=0)
        # conv1_2: Conv2D(25, (5, 1), padding="valid", relu)
        self.conv1_2 = nn.Conv2d(75, 25, kernel_size=(5, 1), padding=0)
        self.relu = nn.ReLU()

        # CuDNNGRU(units=128) over the reshaped (117, 25) sequence.
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)

        self.softmax_fc = nn.Linear(128, classes)

    def forward(self, x, i_chan, q_chan):
        # x: (B, input_len, 2, 1) raw IQ map; i_chan/q_chan: (B, input_len) raw I/Q.
        b = x.shape[0]

        x1 = x.reshape(b, -1)
        x1 = self.fc2(x1)  # linear activation (identity)

        cos1 = torch.cos(x1)  # (B, 1)
        sin1 = torch.sin(x1)  # (B, 1)

        x11 = i_chan * cos1
        x12 = q_chan * sin1
        x21 = q_chan * cos1
        x22 = i_chan * sin1
        y1 = x11 + x12
        y2 = x21 - x22

        y1 = y1.reshape(b, self.input_len, 1)
        y2 = y2.reshape(b, self.input_len, 1)
        x11cat = torch.cat([y1, y2], dim=2)  # (B, input_len, 2)
        x3 = x11cat.reshape(b, 1, self.input_len, 2)  # NCHW: (B, 1, 128, 2)

        x3 = self.relu(self.conv1_1(x3))  # (B, 75, 121, 1)
        x3 = self.relu(self.conv1_2(x3))  # (B, 25, 117, 1)

        x4 = x3.reshape(b, 25, 117).permute(0, 2, 1)  # (B, 117, 25)
        _, h_n = self.gru(x4)
        x4 = h_n[
            -1
        ]  # (B, 128) final GRU hidden state, matches Keras CuDNNGRU(units=128) default output

        out = torch.softmax(self.softmax_fc(x4), dim=-1)
        return out


def build_petcgdnn():
    torch.manual_seed(0)
    model = PETCGDNN(input_len=128, classes=11)
    model.eval()
    return model


def example_input_petcgdnn():
    torch.manual_seed(0)
    batch = 2
    x = torch.randn(batch, 128, 2, 1)
    i_chan = torch.randn(batch, 128)
    q_chan = torch.randn(batch, 128)
    return (x, i_chan, q_chan)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("PET-CGDNN", "build_petcgdnn", "example_input_petcgdnn", 2021, MENAGERIE_ZOO),
]
