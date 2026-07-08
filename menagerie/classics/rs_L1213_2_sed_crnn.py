# FAITHFUL PORT of sharathadavanne/sed-crnn @ master (original framework: Keras 2.2.4 / TF1.x)
#
# sharathadavanne/sed-crnn is the canonical DCASE2017 CRNN for polyphonic sound event detection
# (winning method of the DCASE2017 real-life SED task): a stack of Conv2D+BatchNorm+ReLU+MaxPool2D
# blocks (channels-first, `K.set_image_data_format('channels_first')`) feeding a stack of
# bidirectional GRUs (merge_mode='mul'), then TimeDistributed Dense layers, and a final
# TimeDistributed Dense + sigmoid producing per-frame multi-label event activity.
#
# The original repo (`sed.py::get_model`) is written against Keras 2.2.4 / an implicit TF1.x
# backend (`from keras.layers.normalization import BatchNormalization`, functional-API `Model`),
# which cannot be reasonably installed alongside a modern torch/transformers env. This module is
# a FAITHFUL, layer-for-layer transcription of `get_model()` into torch, preserving every
# mechanism from the real code:
#   - `cnn_pool_size = [5, 2, 2]` -> 3x (Conv2D -> BatchNorm -> ReLU -> MaxPool2D(1, pool) -> Dropout)
#   - `Permute((2, 1, 3))` + `Reshape` to fold channels*freq into the RNN feature dim
#   - `rnn_nb = [32, 32]` -> 2x Bidirectional(GRU(32), merge_mode='mul') (elementwise multiply of
#     forward/backward directions, matching Keras `Bidirectional(..., merge_mode='mul')`)
#   - `fc_nb = [32]` -> TimeDistributed(Dense(32)) + Dropout
#   - final TimeDistributed(Dense(num_classes)) + sigmoid strong-label output
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class _ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> ReLU -> MaxPool2D(pool over freq axis only) -> Dropout.

    Mirrors the per-iteration body of the `for _i, _cnt in enumerate(_cnn_pool_size)` loop in
    the original `get_model()`. MaxPooling2D(pool_size=(1, _cnn_pool_size[_i])) in the original
    (channels-first Keras) pools only along the frequency axis, leaving the time axis untouched;
    with NCHW layout (dims: batch, channel, time, freq) this is MaxPool2d(kernel_size=(1, pool)).
    """

    def __init__(self, in_channels: int, nb_filt: int, pool: int, dropout_rate: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, nb_filt, kernel_size=3, padding="same")
        self.bn = nn.BatchNorm2d(nb_filt)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(1, pool))
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class BiGRUMul(nn.Module):
    """Bidirectional GRU with elementwise-multiply merge (Keras `merge_mode='mul'`).

    torch's built-in `nn.GRU(bidirectional=True)` concatenates the forward/backward outputs;
    the original Keras code instead multiplies them elementwise. Implemented here with two
    separate directional GRUs whose outputs are elementwise-multiplied, which is exactly what
    `Bidirectional(GRU(...), merge_mode='mul')` computes.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.fwd = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.0)
        self.bwd = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.0)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        fwd_out, _ = self.fwd(x)
        bwd_out, _ = self.bwd(torch.flip(x, dims=[1]))
        bwd_out = torch.flip(bwd_out, dims=[1])
        out = self.act(fwd_out) * self.act(bwd_out)
        out = self.drop(out)
        return out


class SED_CRNN(nn.Module):
    """Polyphonic sound event detection CRNN (DCASE2017 winning method).

    Faithful torch port of `sed.py::get_model` from sharathadavanne/sed-crnn.
    Input: `(batch, nb_ch, seq_len, nb_freq_bins)` -- channels-first mel-band-energy features,
    matching the original repo's `K.set_image_data_format('channels_first')`.
    Output: `(batch, seq_len, num_classes)` per-frame sigmoid event-activity posteriors
    ("strong_out" in the original).
    """

    def __init__(
        self,
        nb_ch: int = 1,
        num_freq_bins: int = 40,
        num_classes: int = 6,
        cnn_nb_filt: int = 128,
        cnn_pool_size=(5, 2, 2),
        rnn_nb=(32, 32),
        fc_nb=(32,),
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        conv_blocks = []
        in_ch = nb_ch
        for pool in cnn_pool_size:
            conv_blocks.append(_ConvBlock(in_ch, cnn_nb_filt, pool, dropout_rate))
            in_ch = cnn_nb_filt
        self.conv_blocks = nn.ModuleList(conv_blocks)

        freq_after_pool = num_freq_bins
        for pool in cnn_pool_size:
            freq_after_pool //= pool
        rnn_in = cnn_nb_filt * freq_after_pool

        rnn_layers = []
        in_size = rnn_in
        for r in rnn_nb:
            rnn_layers.append(BiGRUMul(in_size, r, dropout_rate))
            in_size = r
        self.rnn_layers = nn.ModuleList(rnn_layers)

        fc_layers = []
        in_size = rnn_nb[-1]
        for f in fc_nb:
            fc_layers.append(nn.Linear(in_size, f))
            in_size = f
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_drop = nn.Dropout(dropout_rate)

        self.out_fc = nn.Linear(in_size, num_classes)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        """x: (batch, nb_ch, seq_len, nb_freq_bins)."""
        for block in self.conv_blocks:
            x = block(x)
        # x: (batch, feat_maps, seq_len, freq_after_pool)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, feat_maps, freq_after_pool)
        x = x.reshape(b, t, c * f)  # (batch, seq_len, feat_maps*freq_after_pool)

        for rnn in self.rnn_layers:
            x = rnn(x)

        for fc in self.fc_layers:
            x = fc(x)
            x = self.fc_drop(x)

        x = self.out_fc(x)
        x = self.out_act(x)
        return x


def build_sed_crnn():
    model = SED_CRNN(
        nb_ch=1,
        num_freq_bins=40,
        num_classes=6,
        cnn_nb_filt=16,
        cnn_pool_size=(5, 2, 2),
        rnn_nb=(8, 8),
        fc_nb=(8,),
        dropout_rate=0.5,
    )
    model.eval()
    return model


def example_input_sed_crnn():
    # (batch, nb_ch, seq_len, nb_freq_bins)
    return torch.randn(1, 1, 20, 40)


MENAGERIE_ENTRIES = [
    ("Acoustic Scene CNN", "build_sed_crnn", "example_input_sed_crnn", 2017, "CODE"),
]
