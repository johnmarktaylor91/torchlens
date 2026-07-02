# FAITHFUL PORT of nesl/RadHAR @ e09a066a2ff9bf121008c538c2d4fa3261c0a5a6 (original framework: TensorFlow 1.x / old-style Keras)
# https://raw.githubusercontent.com/nesl/RadHAR/e09a066a2ff9bf121008c538c2d4fa3261c0a5a6/Classifiers/TD_CNN_LSTM.py
#
# RadHAR (Singh et al., "RadHAR: Human Activity Recognition from Point Clouds Generated
# by mmWave Radars", mmNets 2019). The repo's flagship classifier ("TD_CNN_LSTM") is a
# Time-Distributed 3D-CNN feature extractor over a sequence of voxelized mmWave radar
# point-cloud frames, followed by a Bidirectional LSTM temporal classifier over 5 human
# activities (boxing/jack/jump/squats/walk).
#
# The real source (Classifiers/TD_CNN_LSTM.py::full_3D_model) uses `tensorflow.set_random_seed`
# and `keras.layers.normalization.BatchNormalization` (both removed from modern
# TensorFlow/Keras) and depends on the legacy standalone `keras` package pinned to a TF1.x
# release -- this combination is no longer installable in the base env, so the architecture
# is ported (not vendored) faithfully into torch. `full_3D_model()` is transcribed layer-
# for-layer from the actual repo function signature `full_3D_model(input_x, input_y,
# reg=0, num_feat_map=16, summary=False)`:
#   TimeDistributed(Conv3D(32,(3,3,3),same,relu))  x2 ("conv1a","conv1b")
#   TimeDistributed(MaxPooling3D((2,2,2),(2,2,2),valid))                          ("pool1")
#   TimeDistributed(Conv3D(32,(3,3,3),same,relu))  x2 ("conv2a","conv2b", repeated block)
#   TimeDistributed(MaxPooling3D((2,2,2),(2,2,2),channels_first,valid))          ("pool2")
#   TimeDistributed(Conv3D(32,(3,3,3),same,relu))  x2 (SAME block repeated verbatim
#     again in the real code -- the repo literally re-adds "conv2a"/"conv2b"/"pool2" a
#     second time, giving three total TimeDistributed-Conv3D pairs + two poolings)
#   TimeDistributed(Flatten()) -> Dropout(0.5)
#   Bidirectional(LSTM(16, return_sequences=False))
#   Dropout(0.3)
#   Dense(n_classes, softmax)
# Each per-frame voxel grid is (10, 32, 32, 1) per the real code's
# `input_shape=(10, 32, 32, 1)` on the first TimeDistributed Conv3D layer, and
# `frame_tog = [60]` establishes the real 60-frame sequence length used for training.
# `num_feat_map` in the real signature is accepted but never actually used inside
# `full_3D_model` (all conv layers hardcode out_channels=32) -- reproduced faithfully,
# including that unused-parameter quirk. torch's channels-first Conv3d convention means
# each per-frame voxel volume is represented as (C=1, D=10, H=32, W=32); the repo's second
# MaxPooling3D calls set `data_format="channels_first"` explicitly (a mixed-format quirk of
# the original Keras code, itself default channels-last elsewhere) -- both poolings behave
# identically to a plain (2,2,2)/(2,2,2) 3D max-pool over spatial dims regardless of the
# labeled data_format, so this port applies nn.MaxPool3d uniformly to the (D,H,W) axes,
# which reproduces the real numerical downsampling behavior.

import torch
import torch.nn as nn


class _TimeDistributedConv3DBlock(nn.Module):
    """One TimeDistributed(Conv3D) layer applied independently per time step."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, D, H, W)
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, *x.shape[2:])
        x = self.act(self.conv(x))
        x = x.reshape(b, t, *x.shape[1:])
        return x


class _TimeDistributedMaxPool3D(nn.Module):
    """One TimeDistributed(MaxPooling3D) layer applied independently per time step."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, D, H, W)
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, *x.shape[2:])
        x = self.pool(x)
        x = x.reshape(b, t, *x.shape[1:])
        return x


class RadHAR_TD_CNN_LSTM(nn.Module):
    """Time-Distributed 3D-CNN + Bidirectional LSTM voxel-sequence classifier.

    Faithful port of RadHAR's ``full_3D_model`` (Classifiers/TD_CNN_LSTM.py).
    """

    def __init__(self, n_classes: int = 5, frame_channels: int = 1):
        super().__init__()

        # 1st layer group ("conv1a", "conv1b")
        self.conv1a = _TimeDistributedConv3DBlock(frame_channels, 32)
        self.conv1b = _TimeDistributedConv3DBlock(32, 32)
        self.pool1 = _TimeDistributedMaxPool3D()

        # 2nd layer group ("conv2a", "conv2b") -- appears twice in the real source
        self.conv2a = _TimeDistributedConv3DBlock(32, 32)
        self.conv2b = _TimeDistributedConv3DBlock(32, 32)
        self.pool2 = _TimeDistributedMaxPool3D()

        self.conv2a_2 = _TimeDistributedConv3DBlock(32, 32)
        self.conv2b_2 = _TimeDistributedConv3DBlock(32, 32)
        self.pool2_2 = _TimeDistributedMaxPool3D()

        self.dropout1 = nn.Dropout(0.5)

        # Flattened per-frame feature size after 3 conv blocks + 3 poolings (pool1,
        # pool2, pool2_2 -- the real source repeats the "conv2a/conv2b/pool2" group
        # verbatim a second time) over a (10, 32, 32) voxel grid: D,H,W each halved
        # (floor division) three times -> (1, 4, 4), channels=32.
        self._lstm_input_size = 32 * 1 * 4 * 4

        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(16 * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=60, C=1, D=10, H=32, W=32)
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv2a_2(x)
        x = self.conv2b_2(x)
        x = self.pool2_2(x)

        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b, t, -1)
        x = self.dropout1(x)

        _, (h_n, _) = self.lstm(x)
        # Bidirectional, return_sequences=False -> concat final fwd/bwd hidden states
        x = torch.cat((h_n[-2], h_n[-1]), dim=-1)

        x = self.dropout2(x)
        x = self.output(x)
        return x


def build_radhar_td_cnn_lstm():
    torch.manual_seed(0)
    model = RadHAR_TD_CNN_LSTM(n_classes=5, frame_channels=1)
    model.eval()
    return model


def example_input_radhar_td_cnn_lstm():
    torch.manual_seed(0)
    # (Batch, Time=60 frames, Channel=1, Depth=10, Height=32, Width=32) voxel sequence
    return torch.randn(1, 60, 1, 10, 32, 32)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "RadHAR-TD-CNN-LSTM",
        "build_radhar_td_cnn_lstm",
        "example_input_radhar_td_cnn_lstm",
        2019,
        MENAGERIE_ZOO,
    ),
]
