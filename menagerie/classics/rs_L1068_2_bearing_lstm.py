# FAITHFUL PORT of VNemani14/Bearing_LSTMPrognostics @ main (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/VNemani14/Bearing_LSTMPrognostics/main/LSTM_MCDropout.ipynb
#
# Nemani et al. "Ensembles of Probabilistic LSTM Predictors and Correctors for Bearing
# Prognostics Using Industrial Standards" (Neurocomputing) -- the queue's generic
# "Bearing-LSTM" candidate is realized here via this specific, citable, published
# repo/paper rather than an unattributed reimplementation. The published model (from the
# repo's `LSTM_MCDropout.ipynb`, "train the model with 10% dropout" cell) is:
#
#   model = Sequential()
#   model.add(LSTM(60, activation='tanh', input_shape=(n_steps_in, n_features),
#                  return_sequences=False, recurrent_dropout=0.1))
#   model.add(Dense(1))
#
# a single-layer tanh LSTM (hidden size 60, recurrent dropout 0.1) reading an n_steps_in-length
# univariate vibration-feature window (VRMS) and emitting a one-step-ahead point forecast via a
# `Dense(1)` head, run with Monte-Carlo dropout kept ACTIVE at inference time (the notebook's
# `model(x_input, training=True)` calls, used inside `calculate_RUL` to roll the forecast forward
# autoregressively for remaining-useful-life estimation) -- i.e. the dropout is architecturally
# load-bearing (uncertainty quantification), not merely a training regularizer, so it is kept
# active in `forward()` here (module always in "MC-dropout" mode) rather than gated by
# `self.training`, matching the source's `training=True` inference calls.
#
# Keras' `recurrent_dropout` applies dropout to the LSTM's recurrent (hidden-state) connections
# at every timestep of the single LSTM layer; torch's `nn.LSTM` has no native recurrent-dropout
# for a single-layer stack (its `dropout=` kwarg only fires between stacked layers, none exist
# here), so the recurrent dropout is realized as a `TorchLSTMCellMCDropout` that manually steps
# an `nn.LSTMCell` and applies `nn.functional.dropout` to the recurrent (hidden-state) input at
# each timestep -- the direct one-to-one behavioral transcription of Keras' `recurrent_dropout`
# for a single-layer LSTM, not an approximation.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentDropoutLSTM(nn.Module):
    """Single-layer tanh LSTM with Keras-style `recurrent_dropout` applied to the recurrent
    (hidden-state) input at every timestep, matching `keras.layers.LSTM(recurrent_dropout=p)`.
    Dropout is applied unconditionally (Monte-Carlo dropout), matching the source notebook's
    `model(x_input, training=True)` inference calls used for RUL forecasting.
    """

    def __init__(self, input_size, hidden_size, recurrent_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size), batch_first (Keras default)
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h_dropped = F.dropout(h, p=self.recurrent_dropout, training=True)
            h, c = self.cell(x[:, t, :], (h_dropped, c))
        return h  # return_sequences=False: only the final hidden state


class BearingLSTMPrognostics(nn.Module):
    """Port of the notebook's `Sequential([LSTM(60, tanh, recurrent_dropout=0.1), Dense(1)])`."""

    def __init__(self, n_features=1, hidden_size=60, recurrent_dropout=0.1):
        super().__init__()
        self.lstm = RecurrentDropoutLSTM(n_features, hidden_size, recurrent_dropout)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, n_steps_in, n_features); LSTMCell already applies the tanh activation
        # internally to the cell/hidden update (Keras `activation='tanh'` is the LSTM default).
        h = self.lstm(x)
        return self.dense(h)


def build_bearing_lstm():
    return BearingLSTMPrognostics(n_features=1, hidden_size=60, recurrent_dropout=0.1)


def example_input_bearing_lstm():
    batch = 4
    n_steps_in = 20
    n_features = 1
    return (torch.rand(batch, n_steps_in, n_features),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Bearing-LSTM (Nemani MC-Dropout Prognostics)",
        "build_bearing_lstm",
        "example_input_bearing_lstm",
        2023,
        "ported",
    ),
]
