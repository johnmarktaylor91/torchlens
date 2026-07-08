# FAITHFUL PORT of https://github.com/smousavi05/CRED @ master (original framework: standalone Keras 2.x / TF1-era)
#
# CRED (Mousavi, Zhu, Ellsworth, Beroza, "CRED: A Deep Residual Network of
# Convolutional and Recurrent Units for Earthquake Signal Detection",
# Scientific Reports 2019) is the confirmed real repo for the "DeepQuake"
# queue candidate (queue notes: "DeepQuake is likely referring to ConvNetQuake
# ... or the seismo-acoustic classification paper; closest confirmed repo is
# smousavi05/CRED"). smousavi05/CRED is the OFFICIAL author repo and its
# `cred_utils.py::model_cred` is the exact architecture used by the paper's
# flagship "cred_original" model (see `cred_original.py`:
# `model = model_cred((151, 41, 3), filters=[8, 16, 32, 64, 128, 256])`).
#
# The repo is standalone Keras 2.x (`from keras.layers import ...`,
# `from keras import backend as K`, `from keras.models import Model`) written
# against TF1-era Keras, not `tensorflow.keras`, and additionally imports
# `obspy` for its data-loading/triggering utilities (irrelevant to the model
# itself but part of `cred_utils.py`). This does not run as-is in a modern
# base env, so per the menagerie rung ladder this is a RUNG-3 FAITHFUL PORT:
# every layer in `model_cred` / `block_CNN` / `block_BiLSTM`
# (cred_utils.py:608-666) is transcribed 1:1 into base-env torch --
#   - stride-2 Conv2D+ReLU downsample -> residual `block_CNN` (BN->ReLU->Conv->
#     BN->ReLU->Conv) added back to the downsample output, TWICE (filters[0],
#     filters[1])
#   - reshape (H, W, C) -> (H, W*C) to flatten the frequency/channel axes into
#     one feature axis per time step
#   - 2-layer residual BiLSTM stack (`block_BiLSTM`: each layer is a
#     Bidirectional LSTM + dropout, summed into the running residual for i>0)
#   - a further uni-directional LSTM + dropout + batchnorm
#   - a TimeDistributed Dense(relu)+BN+dropout head
#   - a final TimeDistributed Dense(1, sigmoid) per-time-step detection head
# No mechanism is invented, reordered, or omitted; only the Keras
# functional-API graph-building style is translated to an eager nn.Module.
#
# Ref: https://github.com/smousavi05/CRED/blob/master/cred_utils.py (block_CNN,
#      block_BiLSTM, model_cred; lines 608-666)
# Ref: https://github.com/smousavi05/CRED/blob/master/cred_original.py (real
#      instantiation: model_cred((151, 41, 3), filters=[8, 16, 32, 64, 128, 256]))

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class BlockCNN(nn.Module):
    """Port of cred_utils.py::block_CNN (residual CNN block).

    Original Keras graph (BN -> ReLU -> Conv -> BN -> ReLU -> Conv), returning
    only the second conv's output (the caller adds it back to the block's
    input to form the residual connection, exactly as in `model_cred`).
    """

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        # Original: Conv2D(filters, (ker-2, ker-2), padding='same')
        k = kernel - 2
        pad = k // 2
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, k, padding=pad)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, k, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x


class BlockBiLSTM(nn.Module):
    """Port of cred_utils.py::block_BiLSTM (residual BiLSTM stack).

    Original Keras graph: for `rnn_depth` layers, apply a Bidirectional LSTM
    (return_sequences=True) + dropout; for i>0, add the new branch back onto
    the running residual (`x = add([x, x_rnn])`), otherwise `x = x_rnn`.
    """

    def __init__(self, input_size: int, hidden_size: int, rnn_depth: int):
        super().__init__()
        self.rnn_depth = rnn_depth
        self.lstms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(rnn_depth):
            in_size = input_size if i == 0 else 2 * hidden_size
            self.lstms.append(nn.LSTM(in_size, hidden_size, batch_first=True, bidirectional=True))
            self.dropouts.append(nn.Dropout(0.7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.rnn_depth):
            x_rnn, _ = self.lstms[i](x)
            x_rnn = self.dropouts[i](x_rnn)
            if i > 0:
                x = x + x_rnn
            else:
                x = x_rnn
        return x


class TimeDistributedDense(nn.Module):
    """Port of Keras `TimeDistributed(Dense(...))`: applies the same Linear
    (+ optional activation) independently at every time step. Equivalent to a
    Linear over the last axis of a (batch, time, features) tensor -- PyTorch's
    nn.Linear already broadcasts over leading dims, so this is a thin,
    faithful wrapper documenting the Keras-side intent (including the
    original `kernel_regularizer=l1(0.01)`, which is a training-time weight
    penalty with no effect on the forward graph and is therefore omitted from
    this inference-only port).
    """

    def __init__(self, in_features: int, out_features: int, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CRED(nn.Module):
    """Faithful port of cred_utils.py::model_cred.

    Original signature: model_cred(shape=(T, F, C), filters=[f0, f1, f2, f3,
    f4, f5]) where the real "cred_original" instantiation uses
    shape=(151, 41, 3), filters=[8, 16, 32, 64, 128, 256]. Only filters[0],
    filters[1], and filters[3] are actually consumed by the graph (filters[2],
    filters[4], filters[5] are declared in the real repo's default config but
    unused by `model_cred`'s body -- verified against cred_utils.py:640-666 --
    so this port keeps that same (unused-tail-of-list) signature shape for
    faithfulness, defaulting the list to the paper's 6-entry filters list).

    Input: (batch, T, F, C) -- a spectrogram/waveform-feature stack, channels
    last as in the original Keras `Input(shape=shape)` (channels-last); this
    port keeps a channels-last input for shape fidelity and transposes to
    NCHW internally for `nn.Conv2d`.
    """

    def __init__(self, in_channels: int = 3, filters=(8, 16, 32, 64, 128, 256)):
        super().__init__()
        f0, f1, _f2, f3 = filters[0], filters[1], filters[2], filters[3]

        # conv2D_2 = Conv2D(filters[0], (9,9), strides=(2,2), padding='same', activation='relu')(inp)
        self.conv2d_2 = nn.Conv2d(in_channels, f0, kernel_size=9, stride=2, padding=4)
        self.act2 = nn.ReLU()
        # res_conv_2 = add([block_CNN(filters[0], 9, conv2D_2), conv2D_2])
        self.block_cnn_2 = BlockCNN(f0, 9)

        # conv2D_3 = Conv2D(filters[1], (5,5), strides=(2,2), padding='same', activation='relu')(res_conv_2)
        self.conv2d_3 = nn.Conv2d(f0, f1, kernel_size=5, stride=2, padding=2)
        self.act3 = nn.ReLU()
        # res_conv_3 = add([block_CNN(filters[1], 5, conv2D_3), conv2D_3])
        self.block_cnn_3 = BlockCNN(f1, 5)

        # reshaped = Reshape((shape[1], shape[2]*shape[3]))(res_conv_3)
        #   -- flattens (freq, channels) into one feature axis per time step;
        #   here freq axis is f1 (channel count after conv2d_3), spatial dims
        #   are folded into the feature axis below at forward() time since
        #   the actual folded size depends on the runtime spatial shape.
        self._f1 = f1

        # res_BIlstm = block_BiLSTM(reshaped, filters=filters[3], rnn_depth=2)
        self.block_bilstm = None  # lazily built once we know the folded feature size
        self._lstm_hidden = f3

        # UNIlstm = LSTM(filters[3], return_sequences=True)(res_BIlstm)
        self.uni_lstm = nn.LSTM(2 * f3, f3, batch_first=True)
        self.uni_dropout = nn.Dropout(0.8)
        self.uni_bn = nn.BatchNorm1d(f3)

        # dense_2 = TimeDistributed(Dense(filters[3], kernel_regularizer=l1(0.01), activation='relu'))(UNIlstm)
        self.dense_2 = TimeDistributedDense(f3, f3, activation=nn.ReLU())
        self.dense_2_bn = nn.BatchNorm1d(f3)
        self.dense_2_dropout = nn.Dropout(0.8)

        # dense_3 = TimeDistributed(Dense(1, kernel_regularizer=l1(0.01), activation='sigmoid'))(dense_2)
        self.dense_3 = TimeDistributedDense(f3, 1, activation=nn.Sigmoid())

    def _build_bilstm_lazy(self, feature_size: int, device, dtype):
        self.block_bilstm = BlockBiLSTM(feature_size, self._lstm_hidden, rnn_depth=2).to(
            device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives channels-last (batch, T, F, C) as in the original
        # Keras Input(shape=(T, F, C)); convert to NCHW for nn.Conv2d.
        x = x.permute(0, 3, 1, 2)

        conv2d_2 = self.act2(self.conv2d_2(x))
        res_conv_2 = self.block_cnn_2(conv2d_2) + conv2d_2

        conv2d_3 = self.act3(self.conv2d_3(res_conv_2))
        res_conv_3 = self.block_cnn_3(conv2d_3) + conv2d_3

        # res_conv_3 is (batch, C=f1, T', F'); Keras reshape folds (F', C)
        # into one feature axis per remaining time step T'.
        b, c, t, f = res_conv_3.shape
        reshaped = res_conv_3.permute(0, 2, 3, 1).reshape(b, t, f * c)

        if self.block_bilstm is None:
            self._build_bilstm_lazy(f * c, reshaped.device, reshaped.dtype)

        res_bilstm = self.block_bilstm(reshaped)

        uni_lstm, _ = self.uni_lstm(res_bilstm)
        uni_lstm = self.uni_dropout(uni_lstm)
        uni_lstm = self.uni_bn(uni_lstm.transpose(1, 2)).transpose(1, 2)

        dense_2 = self.dense_2(uni_lstm)
        dense_2 = self.dense_2_bn(dense_2.transpose(1, 2)).transpose(1, 2)
        dense_2 = self.dense_2_dropout(dense_2)

        dense_3 = self.dense_3(dense_2)
        return dense_3


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). Real-paper defaults
# are input shape (151, 41, 3) with filters=[8, 16, 32, 64, 128, 256] --
# shrunk here to a small spatial/time grid and narrow filter widths for a
# fast CPU trace, same architecture shape (2 residual conv-downsample blocks
# -> reshape -> 2-layer residual BiLSTM -> uni-LSTM -> TimeDistributed dense
# head).
# ---------------------------------------------------------------------------
def build_cred():
    torch.manual_seed(0)
    model = CRED(in_channels=3, filters=(4, 8, 8, 6, 8, 8))
    model.eval()
    # Run one dummy forward pass (eval mode, no_grad) purely to trigger the
    # lazy BiLSTM sub-module build (its input feature size depends on the
    # runtime spatial shape, exactly like Keras' `K.int_shape` static-shape
    # introspection at graph-build time) before the real traced call.
    with torch.no_grad():
        model(torch.zeros(1, 16, 12, 3))
    model.eval()
    return model


def example_input_cred():
    torch.manual_seed(0)
    # (batch, T=16, F=12, C=3) channels-last, matching the real repo's
    # Input(shape=(time, freq, channel)) convention at a shrunk resolution.
    return torch.randn(1, 16, 12, 3)


MENAGERIE_ENTRIES = [
    ("CRED (earthquake signal detection)", "build_cred", "example_input_cred", 2019, MENAGERIE_ZOO),
]
