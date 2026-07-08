# FAITHFUL PORT of YonghaoXu/SSUN @ 6032d05146ab2ccbdc7c51dd37a4bab22fd170fc (original framework: Keras 2.2.4 / TensorFlow 1.13)
# https://raw.githubusercontent.com/YonghaoXu/SSUN/master/SSUN.py
#
# SSUN ("Spectral-Spatial Unified Networks for Hyperspectral Image Classification",
# Xu, Zhang, Du & Zhang, IEEE TGRS 2018). The repo's `SSUN.py` builds the joint model
# via the Keras functional API as two branches fused at multiple points:
#   - Spectral (LSTM) branch: an `LSTM(128)` over the `(time_step, nb_features)` band-
#     group sequence -> `Dense(128, relu)` ("LSTMDense") -> `Dense(nb_classes, softmax)`
#     ("LSTMSOFTMAX").
#   - Spatial (multi-scale CNN, "MSCNN") branch: three stacked `Conv2D(32, 3x3, relu,
#     same) -> MaxPooling2D(2x2)` stages over the `(img_rows, img_cols, num_PC)` patch;
#     EACH pooled stage (POOL1/POOL2/POOL3) is independently flattened and projected
#     through its own `Dense(128, relu)` (DENSE1/DENSE2/DENSE3), and the three 128-dim
#     projections are fused by element-wise `Add()` ("CNNDense") -> `Dense(nb_classes,
#     softmax)` ("CNNSOFTMAX").
#   - Joint head: `Concatenate()([LSTMDense, CNNDense])` -> `Dense(128, relu)`
#     ("JOINTDENSE") -> `Dense(nb_classes, softmax)` ("JOINTSOFTMAX").
# The full `SSUN` model returns all three softmax heads (JOINTSOFTMAX, LSTMSOFTMAX,
# CNNSOFTMAX), exactly as `Model(input=[LSTMInput,CNNInput],
# output=[JOINTSOFTMAX,LSTMSOFTMAX,CNNSOFTMAX])` in the original.
#
# This is a FAITHFUL PORT, not a from-scratch reimplementation: every layer, its
# arguments (channel counts, kernel sizes, activations, fusion points), and the graph
# topology (which pooled stage feeds which dense projection, where Add()/Concatenate()
# happen) are transcribed directly from the real `SSUN()` function above. The legacy
# Keras 2.2.4 `LSTM(..., consume_less='gpu', W_regularizer=..., U_regularizer=...)`
# kwargs are obsolete Keras-1.x-style API (regularizers -> `kernel_regularizer` /
# `recurrent_regularizer`; `consume_less` is a since-removed perf hint with no
# semantic effect on the architecture) and are dropped as non-architectural framework
# cruft; the L2 weight regularization itself is architecture-irrelevant for a forward
# trace (regularizers only affect the loss during training) and is omitted along with
# the training-only `model.compile(...)` call. Training/data-loading/plotting code
# (`HyperspectralSamples`, `CalAccuracy`, `DrawResult`, the `%% Spectral/Spatial/
# Joint` training loops, and `.mat`/`.png` I/O) is dropped as script plumbing, not
# part of the traced architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSUN(nn.Module):
    def __init__(self, time_step, nb_features, num_PC, img_rows, img_cols, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes

        # ---- Spectral (LSTM) branch ----
        # LSTM(128, name='LSTMSpectral') over (time_step, nb_features)
        self.lstm_spectral = nn.LSTM(input_size=nb_features, hidden_size=128, batch_first=True)
        # Dense(128, relu, name='LSTMDense')
        self.lstm_dense = nn.Linear(128, 128)
        # Dense(nb_classes, softmax, name='LSTMSOFTMAX')
        self.lstm_softmax = nn.Linear(128, nb_classes)

        # ---- Spatial (multi-scale CNN) branch ----
        # CONV1/POOL1
        self.conv1 = nn.Conv2d(num_PC, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        # CONV2/POOL2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # CONV3/POOL3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        # spatial dims after each pooling stage (Keras 'same' padding + stride-2 pool,
        # floor division, matching Keras MaxPooling2D default behavior)
        r1, c1 = img_rows // 2, img_cols // 2
        r2, c2 = r1 // 2, c1 // 2
        r3, c3 = r2 // 2, c2 // 2

        # DENSE1/DENSE2/DENSE3 (name='DENSE1' etc.), each Dense(128, relu) over its
        # own flattened pooled-stage feature map
        self.dense1 = nn.Linear(32 * r1 * c1, 128)
        self.dense2 = nn.Linear(32 * r2 * c2, 128)
        self.dense3 = nn.Linear(32 * r3 * c3, 128)

        # Dense(nb_classes, softmax, name='CNNSOFTMAX')
        self.cnn_softmax = nn.Linear(128, nb_classes)

        # ---- Joint head ----
        # Dense(128, relu, name='JOINTDENSE') over Concatenate([LSTMDense, CNNDense])
        self.joint_dense = nn.Linear(128 + 128, 128)
        # Dense(nb_classes, softmax, name='JOINTSOFTMAX')
        self.joint_softmax = nn.Linear(128, nb_classes)

    def forward(self, lstm_input, cnn_input):
        # lstm_input: (batch, time_step, nb_features)
        # cnn_input: (batch, num_PC, img_rows, img_cols)  [NCHW]

        # ---- Spectral branch ----
        _, (h_n, _) = self.lstm_spectral(lstm_input)
        lstm_spectral_out = h_n[-1]  # (batch, 128), last-timestep hidden state
        lstm_dense = F.relu(self.lstm_dense(lstm_spectral_out))
        lstm_softmax = F.softmax(self.lstm_softmax(lstm_dense), dim=-1)

        # ---- Spatial branch ----
        pool1 = self.pool1(F.relu(self.conv1(cnn_input)))
        pool2 = self.pool2(F.relu(self.conv2(pool1)))
        pool3 = self.pool3(F.relu(self.conv3(pool2)))

        flatten1 = pool1.flatten(1)
        flatten2 = pool2.flatten(1)
        flatten3 = pool3.flatten(1)

        dense1 = F.relu(self.dense1(flatten1))
        dense2 = F.relu(self.dense2(flatten2))
        dense3 = F.relu(self.dense3(flatten3))

        cnn_dense = dense1 + dense2 + dense3  # Add()([DENSE1, DENSE2, DENSE3])
        cnn_softmax = F.softmax(self.cnn_softmax(cnn_dense), dim=-1)

        # ---- Joint head ----
        joint = torch.cat([lstm_dense, cnn_dense], dim=-1)  # Concatenate()
        joint_dense = F.relu(self.joint_dense(joint))
        joint_softmax = F.softmax(self.joint_softmax(joint_dense), dim=-1)

        return joint_softmax, lstm_softmax, cnn_softmax


def build_ssun():
    torch.manual_seed(0)
    model = SSUN(
        time_step=3,
        nb_features=6,
        num_PC=4,
        img_rows=16,
        img_cols=16,
        nb_classes=9,
    )
    model.eval()
    return model


def example_input_ssun():
    torch.manual_seed(0)
    lstm_input = torch.randn(2, 3, 6)
    cnn_input = torch.randn(2, 4, 16, 16)
    return lstm_input, cnn_input


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("SSUN", "build_ssun", "example_input_ssun", 2018, MENAGERIE_ZOO),
]
