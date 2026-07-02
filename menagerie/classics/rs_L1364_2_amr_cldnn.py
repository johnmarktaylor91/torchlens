# FAITHFUL PORT of Richardzhangxx/AMR-Benchmark @ main
#   RML201610a/CLDNN/rmlmodels/CLDNNLikeModel.py (original framework: Keras 2.x /
#   TF1.x -- `CuDNNLSTM`, legacy `init=` kwarg, `keras.optimizers.Adam(decay=...)`;
#   this legacy Keras1/TF1-era API is not installable in the current base env, so the
#   architecture is transcribed faithfully into self-contained torch below, layer for
#   layer, rather than vendored).
#
# CLDNN-like automatic modulation recognition (AMC) network from the AMR-Benchmark
# suite (Richardzhangxx/AMR-Benchmark; RadioML2016.10a benchmark for the
# "CLDNN"/"CNN+LSTM+DNN" family used across the AMC literature, e.g. Liu, Yang &
# Wang, 2017; and Rajendran et al.). Input is a raw I/Q radio signal reshaped as a
# (1, 2, 128) "image" (channels_first: 1 channel, 2 = I/Q, 128 = time samples). Three
# stacked ZeroPad+Conv2D(50, (1,8), valid, relu)+Dropout blocks extract local
# time-frequency features (padding only the time axis, kernel spanning only time, so
# the I/Q axis of size 2 is preserved unpooled throughout); the first- and
# third-block feature maps are then channel-preserving concatenated along the time
# axis (`concatenate(axis=-1)` on channels_first (C,H,W) tensors concatenates on W,
# the last positional axis) to skip-connect early and late conv features, reshaped
# to treat the channel dimension as an LSTM timestep axis and run through a single
# CuDNNLSTM(50), then two Dense (ReLU / linear+softmax) layers classify among 11
# modulation classes. Every layer/mechanism is reproduced faithfully from the real
# Keras code; only the framework substrate changes (Keras Conv2D/ZeroPadding2D/
# CuDNNLSTM/Dense -> torch nn.Conv2d/ZeroPad2d/LSTM/Linear).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class CLDNNLikeModel(nn.Module):
    """Faithful port of CLDNNLikeModel(weights=None, input_shape1=[2,128], classes=11).

    Input: (N, 1, 2, 128) channels_first (matches Keras `Input(shape=(1, 2, 128))`).
    """

    def __init__(self, classes=11, dr=0.5):
        super().__init__()
        self.dr = dr

        # ZeroPadding2D((0, 2), channels_first) pads only the last (time/width) axis
        # by 2 on each side; Conv2D(50, (1, 8), valid) then only convolves over time.
        self.conv11 = nn.Conv2d(1, 50, kernel_size=(1, 8), padding=0)
        self.conv12 = nn.Conv2d(50, 50, kernel_size=(1, 8), padding=0)
        self.conv13 = nn.Conv2d(50, 50, kernel_size=(1, 8), padding=0)

        nn.init.xavier_uniform_(self.conv11.weight)
        nn.init.xavier_uniform_(self.conv12.weight)
        nn.init.xavier_uniform_(self.conv13.weight)

        # After the conv stack: layer11 is (N, 50, 2, 125), layer13 is (N, 50, 2, 119)
        # (2 = I/Q axis untouched throughout, 125/119 = time axis after each valid
        # conv). `concatenate([layer11, layer13])` in Keras defaults to axis=-1
        # (the trailing/time axis here) -> concat shape (N, 50, 2, 244).
        # `concat_size[-3]` (=50, the channel axis) becomes the LSTM `timesteps`;
        # `concat_size[-1]*concat_size[-2]` (=244*2=488) becomes the per-timestep
        # feature dim after `Reshape((timesteps, input_dim))`.
        self.lstm_input_dim = 488
        self.lstm_timesteps = 50
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=50, batch_first=True)

        self.dense1 = nn.Linear(50, 256)
        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity="relu")
        self.dense2 = nn.Linear(256, classes)
        nn.init.kaiming_normal_(self.dense2.weight, nonlinearity="relu")

    def forward(self, x):
        """x: (N, 1, 2, 128) channels_first I/Q input."""
        dr = self.dr

        x_pad = F.pad(x, (2, 2, 0, 0))  # ZeroPadding2D((0, 2))
        layer11 = F.relu(self.conv11(x_pad))  # (N, 50, 2, 125)
        layer11 = F.dropout(layer11, p=dr, training=self.training)

        layer11_pad = F.pad(layer11, (2, 2, 0, 0))
        layer12 = F.relu(self.conv12(layer11_pad))  # (N, 50, 2, 122)
        layer12 = F.dropout(layer12, p=dr, training=self.training)

        layer12_pad = F.pad(layer12, (2, 2, 0, 0))
        layer13 = F.relu(self.conv13(layer12_pad))  # (N, 50, 2, 119)
        layer13 = F.dropout(layer13, p=dr, training=self.training)

        concat = torch.cat([layer11, layer13], dim=-1)  # (N, 50, 2, 244), axis=-1 (time)

        n = concat.shape[0]
        # Reshape((timesteps, input_dim)): timesteps = channel axis (50),
        # input_dim = H*W flattened (2*244=488). Keras Reshape uses row-major
        # (C-order) flattening of the trailing dims, matching torch's default
        # .reshape/.view semantics on a contiguous tensor.
        concat = concat.reshape(n, self.lstm_timesteps, self.lstm_input_dim)

        lstm_out, _ = self.lstm(concat)
        lstm_out = lstm_out[:, -1, :]  # CuDNNLSTM returns only the final hidden state

        layer_dense1 = F.relu(self.dense1(lstm_out))
        layer_dropout = F.dropout(layer_dense1, p=dr, training=self.training)
        layer_dense2 = self.dense2(layer_dropout)
        output = F.softmax(layer_dense2, dim=-1)
        return output


# ---- staging entry points ----


def build_amr_cldnn():
    torch.manual_seed(0)
    model = CLDNNLikeModel(classes=11)
    model.eval()
    return model


def example_input_amr_cldnn():
    torch.manual_seed(0)
    return (torch.randn(2, 1, 2, 128),)


MENAGERIE_ENTRIES = [
    ("AMR-CLDNN", "build_amr_cldnn", "example_input_amr_cldnn", 2020, MENAGERIE_ZOO),
]
