# FAITHFUL PORT of https://github.com/nesl/RadHAR @ master (original framework: TF1.x/Keras)
#
# RadHAR (Singh, Sandha, Garcia, Srivastava. 2019, "RadHAR: Human Activity Recognition
# from Point Clouds Generated through a Millimeter-wave Radar", mmNets @ MobiCom 2019).
# https://github.com/nesl/RadHAR
#
# The real architecture is `full_3D_model()` in `Classifiers/TD_CNN_LSTM.py`:
#   https://raw.githubusercontent.com/nesl/RadHAR/master/Classifiers/TD_CNN_LSTM.py
#
# That file targets `tensorflow.set_random_seed` (removed in TF2) and `keras.layers
# .normalization.BatchNormalization` / old standalone `keras` (pre-TF2 Keras package
# layout) -- a TF1.x-era environment that cannot be reasonably installed alongside our
# current torch/torchvision/etc. base env, so per the ladder this is a rung-3 faithful
# TRANSCRIPTION of the real `full_3D_model()` Sequential stack into self-contained torch:
#
#   model.add(TimeDistributed(Conv3D(32, (3,3,3), strides=(1,1,1), padding="same", activation="relu")))  x2
#   model.add(TimeDistributed(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")))
#   model.add(TimeDistributed(Conv3D(32, (3,3,3), strides=(1,1,1), padding="same", activation="relu")))  x2
#   model.add(TimeDistributed(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")))
#   model.add(TimeDistributed(Conv3D(32, (3,3,3), strides=(1,1,1), padding="same", activation="relu")))  x2
#   model.add(TimeDistributed(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid")))
#   model.add(TimeDistributed(Flatten()))
#   model.add(Dropout(0.5))
#   model.add(Bidirectional(LSTM(16, return_sequences=False, stateful=False)))
#   model.add(Dropout(0.3))
#   model.add(Dense(num_classes, activation='softmax'))
#
# Every mechanism is preserved: `TimeDistributed(Conv3D)`/`TimeDistributed(MaxPooling3D)`
# are implemented by folding the time axis into the batch axis before each per-frame op
# and unfolding it back (the exact semantics of Keras's TimeDistributed wrapper); the
# `padding="same"` 3x3x3 convs use `padding=1` (identical spatial-preserving effect for
# odd kernels/stride 1); `Bidirectional(LSTM(..., return_sequences=False))` is
# `nn.LSTM(bidirectional=True)` with the two DIRECTIONS' final hidden states
# concatenated (`torch.cat([h_n[0], h_n[1]], dim=-1)`) -- this is the real Keras
# behavior of a non-sequence-returning bidirectional wrapper (forward direction's last
# processed step concatenated with the backward direction's last processed step, i.e.
# the reversed sequence's own last step), NOT `out[:, -1, :]` of the full output
# sequence (which would silently drop the backward branch's actual final state).
# `Dropout` layers are kept as real `nn.Dropout` modules (inert in eval-mode trace, as
# in Keras inference too).
#
# The real file also intermixes NumPy data-loading, `sklearn.train_test_split`, model
# compile/`fit`/`ModelCheckpoint` training-loop code with the model definition; none of
# that is part of the trainable network and is not ported (matching menagerie's existing
# convention of extracting only the architecture-relevant slice; see e.g. AD-MLP).
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class TDCnnLstm(nn.Module):
    """Faithful port of RadHAR's `full_3D_model()` (TD_CNN_LSTM.py): a
    TimeDistributed 3D-CNN voxel-sequence encoder followed by a bidirectional LSTM
    classification head."""

    def __init__(
        self,
        voxel_size: int = 32,
        num_feat_map: int = 32,
        num_classes: int = 5,
        lstm_hidden: int = 16,
    ):
        super().__init__()
        self.conv1a = nn.Conv3d(1, num_feat_map, 3, stride=1, padding=1)
        self.conv1b = nn.Conv3d(num_feat_map, num_feat_map, 3, stride=1, padding=1)
        self.conv2a = nn.Conv3d(num_feat_map, num_feat_map, 3, stride=1, padding=1)
        self.conv2b = nn.Conv3d(num_feat_map, num_feat_map, 3, stride=1, padding=1)
        self.conv3a = nn.Conv3d(num_feat_map, num_feat_map, 3, stride=1, padding=1)
        self.conv3b = nn.Conv3d(num_feat_map, num_feat_map, 3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        # 3 halving MaxPool3D(2,2,2) stages: voxel_size -> voxel_size // 8
        spatial = voxel_size // 8
        flat_dim = num_feat_map * spatial * spatial * spatial
        self.lstm = nn.LSTM(
            input_size=flat_dim, hidden_size=lstm_hidden, batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.3)
        self.dense = nn.Linear(lstm_hidden * 2, num_classes)

    def _time_distributed_conv(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        b, t, c, d, h, w = x.shape
        x = x.reshape(b * t, c, d, h, w)
        x = self.relu(conv(x))
        _, c2, d2, h2, w2 = x.shape
        return x.reshape(b, t, c2, d2, h2, w2)

    def _time_distributed_pool(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = x.shape
        x = x.reshape(b * t, c, d, h, w)
        x = self.pool(x)
        _, c2, d2, h2, w2 = x.shape
        return x.reshape(b, t, c2, d2, h2, w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels=1, D, H, W) voxel grid sequence.
        x = self._time_distributed_conv(x, self.conv1a)
        x = self._time_distributed_conv(x, self.conv1b)
        x = self._time_distributed_pool(x)

        x = self._time_distributed_conv(x, self.conv2a)
        x = self._time_distributed_conv(x, self.conv2b)
        x = self._time_distributed_pool(x)

        x = self._time_distributed_conv(x, self.conv3a)
        x = self._time_distributed_conv(x, self.conv3b)
        x = self._time_distributed_pool(x)

        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b, t, -1)  # TimeDistributed(Flatten())
        x = self.dropout1(x)

        # Bidirectional(LSTM(..., return_sequences=False)): concatenate the forward
        # direction's final hidden state with the backward direction's final hidden
        # state (h_n[0]=forward last step, h_n[1]=backward last-processed step).
        _, (h_n, _) = self.lstm(x)
        x = torch.cat([h_n[0], h_n[1]], dim=-1)

        x = self.dropout2(x)
        x = self.dense(x)
        return torch.softmax(x, dim=-1)


def build_radhar():
    # Tiny config: smaller voxel grid / feature width / LSTM hidden size than the
    # paper's real 32^3 voxels + 32 feature maps, but every architectural stage
    # (2x TD-Conv3D + TD-MaxPool3D per level x3 levels, bidirectional LSTM head) is
    # exercised exactly as in the real Sequential stack.
    return TDCnnLstm(voxel_size=16, num_feat_map=4, num_classes=5, lstm_hidden=4)


def example_input_radhar():
    # (batch, 10 time windows, 1 channel, D, H, W) voxelized point-cloud sequence,
    # matching the real code's `train_data.reshape(N, frames, 32, 32, 32, 1)` layout
    # (channels-first here instead of Keras's channels-last).
    return torch.randn(2, 10, 1, 16, 16, 16)


MENAGERIE_ENTRIES = [
    ("RadHAR", "build_radhar", "example_input_radhar", 2019, "ported-pytorch"),
]
