# FAITHFUL PORT of sharathadavanne/seld-dcase2021 @ 24d14c9ebe6878062dc3b68eb07dd0100722f530
# (original framework: Keras 1.x/TF1-era `keras.layers.core`/`keras.layers.recurrent`/
# `keras.layers.wrappers` API -- these submodules were removed from modern Keras/TF and
# the repo does not run against any installed base lib here)
# https://raw.githubusercontent.com/sharathadavanne/seld-dcase2021/master/keras_model.py
#
# SPADE-SED (the sharathadavanne/seld-dcase2021 DCASE2021-Task3 baseline "SELDnet"
# architecture -- distinct from the 2023 CRNN+self-attention SeldModel already vendored
# from seld-dcase2023 in this menagerie, since this 2021 baseline has no self-attention
# stage and produces two heads, SED + DOA, rather than a single ACCDOA head): a stack of
# Conv2D(3x3,same)+BatchNorm2d+ReLU+MaxPool2d(t_pool,f_pool)+Dropout blocks over a
# (mic_channels, time, mel) spectrogram, a channels-last permute + reshape to fold the
# frequency axis into the feature axis, a stack of bidirectional GRU layers, a per-timestep
# FC "DOA" branch (TimeDistributed Dense+Dropout stack -> Dense -> tanh) and a per-timestep
# FC "SED" branch (TimeDistributed Dense+Dropout stack -> Dense -> sigmoid) both fed from
# the same GRU output. `get_model`'s Conv2D/BatchNormalization/MaxPooling2D/Dropout/
# Bidirectional-GRU/TimeDistributed-Dense stack is transcribed layer-for-layer into
# `nn.Conv2d`/`nn.BatchNorm2d`/`nn.MaxPool2d`/`nn.Dropout`/`nn.GRU(bidirectional=True)`/
# `nn.Linear` (a `TimeDistributed(Dense)` in Keras is exactly a `nn.Linear` applied over
# the trailing feature dim of a [B, T, F] tensor -- no PyTorch wrapper needed). Only the
# `masked_mse` loss / `Model.compile` / training machinery (not part of the traced
# architecture) is dropped; this port uses the `is_accdoa=False`, `doa_objective='mse'`
# branch (dual SED+DOA output, the more architecturally complete branch of `get_model`).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ConvBlock(nn.Module):
    """Conv2D(3x3,padding='same') + BatchNormalization + Activation('relu') +
    MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i])) + Dropout(dropout_rate),
    one iteration of the `for i, convCnt in enumerate(f_pool_size)` loop body in
    the original `get_model`."""

    def __init__(self, in_channels, out_channels, t_pool_size, f_pool_size, dropout_rate):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(t_pool_size, f_pool_size))
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class SpadeSED(nn.Module):
    """Faithful port of `keras_model.get_model(..., is_accdoa=False)`.

    Args mirror the Keras `get_model` signature: `data_in = (batch, mic_channels,
    time_steps, mel_bins)` (channels_first, matching the repo's
    `keras.backend.set_image_data_format('channels_first')`), `nb_cnn2d_filt` /
    `f_pool_size` / `t_pool_size` control the CNN stack, `rnn_size` is a list of
    bidirectional-GRU widths, `fnn_size` is a list of FC widths shared by both the
    SED and DOA output heads, `n_sed_classes` is the per-timestep SED output width
    (`data_out[0][-1]`), and `n_doa_outputs` is the per-timestep DOA output width
    (`data_out[1][-1]`).
    """

    def __init__(
        self,
        data_in,
        nb_cnn2d_filt,
        f_pool_size,
        t_pool_size,
        dropout_rate,
        rnn_size,
        fnn_size,
        n_sed_classes,
        n_doa_outputs,
    ):
        super().__init__()
        _batch, mic_channels, _time_steps, mel_bins = data_in

        # CNN stack
        conv_blocks = []
        in_ch = mic_channels
        freq = mel_bins
        for i in range(len(f_pool_size)):
            conv_blocks.append(
                ConvBlock(in_ch, nb_cnn2d_filt, t_pool_size[i], f_pool_size[i], dropout_rate)
            )
            in_ch = nb_cnn2d_filt
            freq = freq // f_pool_size[i]
        self.conv_block_list = nn.ModuleList(conv_blocks)

        # after `Permute((2, 1, 3))` (channels_first NCHW -> N, time, chn, freq) and
        # `Reshape((time_steps_pooled, -1))`, the GRU input feature dim is chn * freq
        gru_input_dim = nb_cnn2d_filt * freq

        gru_layers = []
        in_dim = gru_input_dim
        for rnn_filt in rnn_size:
            gru_layers.append(
                nn.GRU(
                    input_size=in_dim, hidden_size=rnn_filt, batch_first=True, bidirectional=True
                )
            )
            # Bidirectional(merge_mode='mul'): elementwise-multiply fwd/bwd directions,
            # so the feature width feeding the next GRU layer stays `rnn_filt`
            in_dim = rnn_filt
        self.gru_list = nn.ModuleList(gru_layers)

        # DOA head: TimeDistributed(Dense) + Dropout stack, then a final Dense -> tanh
        doa_fc = []
        in_dim = rnn_size[-1] if rnn_size else gru_input_dim
        for fnn_filt in fnn_size:
            doa_fc.append(nn.Linear(in_dim, fnn_filt))
            in_dim = fnn_filt
        self.doa_fc_list = nn.ModuleList(doa_fc)
        self.doa_dropout = nn.Dropout(dropout_rate)
        self.doa_out = nn.Linear(in_dim, n_doa_outputs)

        # SED head: mirrors the DOA head structure, independent weights
        sed_fc = []
        in_dim = rnn_size[-1] if rnn_size else gru_input_dim
        for fnn_filt in fnn_size:
            sed_fc.append(nn.Linear(in_dim, fnn_filt))
            in_dim = fnn_filt
        self.sed_fc_list = nn.ModuleList(sed_fc)
        self.sed_dropout = nn.Dropout(dropout_rate)
        self.sed_out = nn.Linear(in_dim, n_sed_classes)

    def forward(self, x):
        """x: [Batch, mic_channels, time_steps, mel_bins] (channels_first)."""
        for block in self.conv_block_list:
            x = block(x)
        # Permute((2, 1, 3)): [B, C, T, F] -> [B, T, C, F]
        x = x.permute(0, 2, 1, 3).contiguous()
        # Reshape((time_steps_pooled, -1)): [B, T, C, F] -> [B, T, C*F]
        x = x.reshape(x.shape[0], x.shape[1], -1)

        for gru in self.gru_list:
            x, _ = gru(x)
            # Bidirectional(..., merge_mode='mul'): split fwd/bwd halves, multiply
            half = x.shape[-1] // 2
            x = torch.tanh(x)
            x = x[..., :half] * x[..., half:]

        doa = x
        for fc in self.doa_fc_list:
            doa = fc(doa)
            doa = self.doa_dropout(doa)
        doa = torch.tanh(self.doa_out(doa))

        sed = x
        for fc in self.sed_fc_list:
            sed = fc(sed)
            sed = self.sed_dropout(sed)
        sed = torch.sigmoid(self.sed_out(sed))

        return sed, doa


def build_spade_sed():
    model = SpadeSED(
        data_in=(2, 4, 8, 16),  # (batch, mic_channels, time_steps, mel_bins)
        nb_cnn2d_filt=8,
        f_pool_size=[2, 2],
        t_pool_size=[1, 1],
        dropout_rate=0.0,
        rnn_size=[16],
        fnn_size=[16],
        n_sed_classes=3,
        n_doa_outputs=9,
    )
    model.eval()
    return model


def example_input_spade_sed():
    return torch.randn(2, 4, 8, 16)


MENAGERIE_ENTRIES = [
    ("SPADE-SED", "build_spade_sed", "example_input_spade_sed", 2021, MENAGERIE_ZOO),
]
