# FAITHFUL PORT of https://github.com/flaviagiammarino/encdec-ad-tensorflow @ main
# (encdec_ad_tensorflow/modules.py, `EncoderDecoder` class) (original
# framework: TensorFlow / Keras)
#
# EncDec-AD (Malhotra et al., ICML 2016 Anomaly Detection Workshop,
# "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"):
# an LSTM encoder compresses an input window of length L into a final
# (hidden, cell) state; an LSTMCell decoder then reconstructs the window
# IN REVERSE ORDER, one step at a time, starting from the encoder's final
# state. Each decoder step's hidden state passes through a shared Dense
# ("outputs") readout to produce the reconstruction at that timestep. The
# real repo's `training` flag switches the decoder's per-step input between
# TEACHER FORCING (the true input at t+1) and self-feeding (the model's own
# just-produced reconstruction at t+1) -- both branches are faithfully
# reproduced below via the same `training: bool` argument to `forward`.
#
# Only the pure Keras -> torch layer mapping was translated (`tf.keras.layers
# .LSTM(..., return_state=True)` -> `torch.nn.LSTM`; `tf.keras.layers.
# LSTMCell` -> `torch.nn.LSTMCell`; `tf.keras.layers.Dense` -> `torch.nn.
# Linear`; the manual `tf.TensorArray` write-loop -> an equivalent Python
# list + `torch.stack`). No mechanism (encoder-then-reverse-decode,
# teacher-forcing switch, shared readout head) was added, removed, or
# altered. `model.py`'s scaling/windowing/F-beta-threshold training-and-
# scoring scaffolding is not part of the architecture and was dropped.

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class EncoderDecoder(nn.Module):
    """LSTM encoder -> reverse-order LSTMCell decoder reconstruction head,
    ported line-for-line from the real repo's `EncoderDecoder.call`.

    Args:
        L: input/reconstruction sequence length.
        m: number of input features (sensors).
        c: LSTM hidden size.
        d: dropout rate (applied to the encoder LSTM and to the decoder
            cell's input/recurrent connections, matching Keras `dropout=d`
            on `LSTM`/`LSTMCell`).
    """

    def __init__(self, L, m, c, d=0.0):
        super().__init__()
        self.L = L
        self.m = m
        self.c = c

        # tf.keras.layers.LSTM(units=c, dropout=d, return_state=True)
        self.encoder = nn.LSTM(input_size=m, hidden_size=c, batch_first=True, dropout=0.0)
        self._encoder_input_dropout = nn.Dropout(p=d)

        # tf.keras.layers.LSTMCell(units=c, dropout=d)
        self.decoder = nn.LSTMCell(input_size=m, hidden_size=c)
        self._decoder_input_dropout = nn.Dropout(p=d)

        # tf.keras.layers.Dense(units=m)
        self.outputs = nn.Linear(c, m)

    def forward(self, inputs, training=True):
        # inputs: (batch, L, m)
        enc_in = self._encoder_input_dropout(inputs) if self.training else inputs
        _, (he, ce) = self.encoder(enc_in)
        # nn.LSTM returns (num_layers, batch, hidden); squeeze the 1 layer
        hd = he.squeeze(0).clone()
        cd = ce.squeeze(0).clone()

        r = [None] * self.L

        # r[L-1] = outputs(hd)  (decoder's first produced step, matching the
        # real repo writing index=self.L - 1 before the reverse loop starts)
        r[self.L - 1] = self.outputs(hd)

        for t in range(self.L - 2, -1, -1):
            if training:
                step_in = inputs[:, t + 1, :]
            else:
                step_in = r[t + 1]
            step_in = self._decoder_input_dropout(step_in) if self.training else step_in

            hd, cd = self.decoder(step_in, (hd, cd))
            r[t] = self.outputs(hd)

        return torch.stack(r, dim=1)  # (batch, L, m)


def build_encdecad():
    # L=6 timesteps, m=3 sensors, c=8 hidden units, d=0 dropout (matches the
    # real repo's default d=0 and keeps the tiny-config trace deterministic).
    return EncoderDecoder(L=6, m=3, c=8, d=0.0)


def example_input_encdecad():
    batch = 2
    L, m = 6, 3
    x = torch.randn(batch, L, m)
    return (x, True)  # (inputs, training=True) -> exercises the teacher-forcing path


MENAGERIE_ENTRIES = [
    (
        "EncDec-AD",
        build_encdecad,
        example_input_encdecad,
        2016,
        MENAGERIE_ZOO,
    ),
]
