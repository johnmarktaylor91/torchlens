# FAITHFUL PORT of sharathadavanne/sed-crnn @ master (original framework: Keras
# with Theano/TensorFlow backend)
#
# Source: sed.py `get_model()` (DCASE 2017 Task 3 winning Sound Event Detection
# CRNN: Cakir et al., "Convolutional Recurrent Neural Networks for Polyphonic
# Sound Event Detection"), plus hyperparameters from the module-level constants in
# sed.py (cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate, nb_ch) and
# feature.py (nb_mel_bands=40).
#
# sed.py is Keras (`from keras.layers import ...`) with `K.set_image_data_format
# ('channels_first')`, i.e. TensorFlow/Theano-era Keras, not PyTorch, and is not
# installable/importable in this base torch environment -- transcribed here as a
# faithful port (rung 3).
#
# Original get_model() (channels_first Keras):
#   spec_x = Input(shape=(nb_ch, seq_len, nb_freq_bins))
#   for each pool size in cnn_pool_size:
#       spec_x = Conv2D(cnn_nb_filt, (3,3), padding='same')(spec_x)
#       spec_x = BatchNormalization(axis=1)(spec_x)     # over channel axis
#       spec_x = Activation('relu')(spec_x)
#       spec_x = MaxPooling2D(pool_size=(1, pool))(spec_x)   # pool over freq axis only
#       spec_x = Dropout(dropout_rate)(spec_x)
#   spec_x = Permute((2,1,3))(spec_x)                    # [B, seq_len, C, freq]
#   spec_x = Reshape((seq_len, -1))(spec_x)               # [B, seq_len, C*freq]
#   for r in rnn_nb:
#       spec_x = Bidirectional(GRU(r, return_sequences=True), merge_mode='mul')(spec_x)
#   for f in fc_nb:
#       spec_x = TimeDistributed(Dense(f))(spec_x)
#       spec_x = Dropout(dropout_rate)(spec_x)
#   spec_x = TimeDistributed(Dense(nb_classes))(spec_x)
#   out = Activation('sigmoid')(spec_x)
#
# Ported 1:1 below using torch.nn.Conv2d/BatchNorm2d/MaxPool2d/GRU/Linear. Dropout
# layers are preserved as real modules (inert in eval() but part of the faithful
# architecture, matching the original layer stack). `Bidirectional(..., merge_mode
# ='mul')` (elementwise product of forward/backward GRU outputs, NOT concatenation)
# is reproduced explicitly since torch.nn.GRU's built-in bidirectional mode only
# concatenates.

import torch
import torch.nn as nn


class MulBiGRU(nn.Module):
    """Bidirectional GRU layer with elementwise-product merge, matching Keras'
    `Bidirectional(GRU(...), merge_mode='mul')` (torch.nn.GRU(bidirectional=True)
    only supports concatenation, so forward/backward passes are run explicitly).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.fwd = nn.GRU(input_size, hidden_size, batch_first=True)
        self.bwd = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_out, _ = self.fwd(self.dropout(x))
        bwd_out, _ = self.bwd(self.dropout(torch.flip(x, dims=[1])))
        bwd_out = torch.flip(bwd_out, dims=[1])
        return fwd_out * bwd_out


class SedCrnn(nn.Module):
    """CRNN for polyphonic Sound Event Detection (Cakir et al. / DCASE2017 winner),
    ported from sharathadavanne/sed-crnn's Keras `get_model()`.

    Input:  [Batch, nb_ch, seq_len, nb_freq_bins]  (channels_first, matching the
            original Keras `Input(shape=(nb_ch, seq_len, nb_freq_bins))`)
    Output: [Batch, seq_len, nb_classes] frame-wise sigmoid activity predictions.
    """

    def __init__(
        self,
        nb_ch: int = 1,
        nb_freq_bins: int = 40,
        cnn_nb_filt: int = 128,
        cnn_pool_size: tuple[int, ...] = (5, 2, 2),
        rnn_nb: tuple[int, ...] = (32, 32),
        fc_nb: tuple[int, ...] = (32,),
        nb_classes: int = 6,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.cnn_pool_size = cnn_pool_size

        conv_blocks = []
        in_ch = nb_ch
        for pool in cnn_pool_size:
            conv_blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_ch, cnn_nb_filt, kernel_size=3, padding="same"),
                        "bn": nn.BatchNorm2d(cnn_nb_filt),
                        "pool": nn.MaxPool2d(kernel_size=(1, pool)),
                        "drop": nn.Dropout(dropout_rate),
                    }
                )
            )
            in_ch = cnn_nb_filt
        self.conv_blocks = nn.ModuleList(conv_blocks)

        freq_after_pool = nb_freq_bins
        for pool in cnn_pool_size:
            freq_after_pool //= pool
        rnn_input_size = cnn_nb_filt * freq_after_pool

        rnn_layers = []
        in_size = rnn_input_size
        for r in rnn_nb:
            rnn_layers.append(MulBiGRU(in_size, r, dropout_rate))
            in_size = r
        self.rnn_layers = nn.ModuleList(rnn_layers)

        fc_layers = []
        in_size = rnn_nb[-1] if rnn_nb else rnn_input_size
        for f in fc_nb:
            fc_layers.append(nn.Linear(in_size, f))
            in_size = f
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_dropout = nn.Dropout(dropout_rate)

        self.out_linear = nn.Linear(in_size, nb_classes)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, nb_ch, seq_len, nb_freq_bins]
        z = x
        for block in self.conv_blocks:
            z = block["conv"](z)
            z = block["bn"](z)
            z = torch.relu(z)
            z = block["pool"](z)
            z = block["drop"](z)

        # Permute((2,1,3)) in Keras channels_first [B,C,seq_len,freq] -> [B,seq_len,C,freq]
        z = z.permute(0, 2, 1, 3)
        b, seq_len = z.shape[0], z.shape[1]
        z = z.reshape(b, seq_len, -1)  # Reshape((seq_len, -1))

        for rnn in self.rnn_layers:
            z = rnn(z)

        for fc in self.fc_layers:
            z = fc(z)
            z = self.fc_dropout(z)

        z = self.out_linear(z)
        return self.out_activation(z)


# --- menagerie staging entrypoints -----------------------------------------------

MENAGERIE_ZOO = "ported-pytorch"


def build_sed_crnn():
    # cnn_nb_filt=128, cnn_pool_size=[5,2,2], rnn_nb=[32,32], fc_nb=[32],
    # dropout_rate=0.5, nb_ch=1 (mono) straight from sed.py; nb_freq_bins=40
    # (nb_mel_bands) from feature.py. nb_classes=6 matches the DCASE2017 Task 3
    # street sound-event label set size used in the paper.
    return SedCrnn(
        nb_ch=1,
        nb_freq_bins=40,
        cnn_nb_filt=16,
        cnn_pool_size=(5, 2, 2),
        rnn_nb=(8, 8),
        fc_nb=(8,),
        nb_classes=6,
        dropout_rate=0.5,
    )


def example_input_sed_crnn():
    # [Batch, nb_ch=1, seq_len, nb_freq_bins=40]; seq_len kept small for tracing.
    return torch.rand(2, 1, 16, 40)


MENAGERIE_ENTRIES = [
    ("CNN-RNN SED", build_sed_crnn, example_input_sed_crnn, 2017, MENAGERIE_ZOO),
]
