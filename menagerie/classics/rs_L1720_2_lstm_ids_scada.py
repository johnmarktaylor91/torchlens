# FAITHFUL PORT of pwwl/ics-anomaly-detection @ 30d4439 (original framework:
# TensorFlow/Keras)
#
# Ported from: detector/lstm.py, class LongShortTermMemory.create_model()
#
# LSTM-IDS-SCADA: the LSTM forecasting-based intrusion/anomaly detector for ICS/SCADA
# telemetry (SWaT/WADI) from the `ics-anomaly-detection` benchmark toolkit (Fung et al.).
# Detection is done by feeding a history window of sensor readings through a stacked LSTM
# forecaster and flagging points whose reconstruction/forecast error exceeds a threshold
# -- the exact architecture this queue entry ("LSTM AE on SWaT/SCADA") maps to in the
# repo's `detector/` package (there is no separate top-level `models/` dir; the real
# detector classes live under `detector/`).
#
# Original Keras graph (transcribed layer-for-layer from `create_model`, default params
# nI=<n_features>, units=64, history=50, layers=2, activation='tanh'):
#
#   input_layer = Input(shape=(history, nI))
#   # layers > 1: first LSTM(s) return_sequences=True, LAST LSTM does not
#   lstmlayer = LSTM(units, activation='tanh', dropout=0.5, return_sequences=True)(input)
#   for _ in range(layers - 2):
#       lstmlayer = LSTM(units, activation='tanh', dropout=0.5, return_sequences=True)(lstmlayer)
#   lstmlayer = LSTM(units)(lstmlayer)            # final layer: return_sequences=False
#   dense_out = Dense(nI)(lstmlayer)              # forecast next timestep, no activation
#   model = Model(input_layer, dense_out)
#
# (the `layers == 1` single-LSTM-directly-on-input branch is also preserved below).
# Keras `LSTM(..., dropout=0.5)` applies dropout to the layer's *input* connections at
# every timestep (a training-time regularizer baked into the recurrent layer itself,
# distinct from a separate `Dropout` module); it is represented here with an explicit
# `nn.Dropout` applied to each LSTM layer's input to keep forward-graph parity, since
# torch's `nn.LSTM` has no built-in input-dropout equivalent.

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class LSTMForecastDetector(nn.Module):
    """Faithful torch port of detector.lstm.LongShortTermMemory.create_model(): a
    stacked LSTM forecaster (intermediate layers return full sequences, the final
    layer returns only the last hidden state) followed by a linear projection back
    to the sensor-feature dimensionality (the one-step-ahead forecast)."""

    def __init__(self, n_features, units=64, history=50, layers=2, dropout=0.5):
        super().__init__()
        assert layers >= 1, "Must have at least one layer."
        self.layers = layers
        self.history = history

        self.lstm_layers = nn.ModuleList()
        self.input_dropouts = nn.ModuleList()

        if layers > 1:
            # first layer: input_size = n_features, return_sequences=True
            self.input_dropouts.append(nn.Dropout(dropout))
            self.lstm_layers.append(
                nn.LSTM(input_size=n_features, hidden_size=units, batch_first=True)
            )
            # middle layers (layers - 2 of them): return_sequences=True
            for _ in range(layers - 2):
                self.input_dropouts.append(nn.Dropout(dropout))
                self.lstm_layers.append(
                    nn.LSTM(input_size=units, hidden_size=units, batch_first=True)
                )
            # final layer: return_sequences=False
            self.input_dropouts.append(nn.Dropout(dropout))
            self.lstm_layers.append(nn.LSTM(input_size=units, hidden_size=units, batch_first=True))
        else:
            # single layer connects directly to input, return_sequences=False
            self.input_dropouts.append(nn.Dropout(dropout))
            self.lstm_layers.append(
                nn.LSTM(input_size=n_features, hidden_size=units, batch_first=True)
            )

        self.dense_out = nn.Linear(units, n_features)

    def forward(self, x):
        # x: (batch, history, n_features)
        out = x.float()
        for i, (drop, lstm) in enumerate(zip(self.input_dropouts, self.lstm_layers)):
            out = drop(out)
            out, _ = lstm(out)
            is_last = i == len(self.lstm_layers) - 1
            if is_last:
                out = out[:, -1, :]  # return_sequences=False on the final layer
        return self.dense_out(out)


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_N_FEATURES = 6  # SWaT/WADI have dozens of sensor channels; shrunk for tracing
_UNITS = 16  # default units=64, shrunk for tracing
_HISTORY = 12  # default history=50, shrunk for tracing
_LAYERS = 2
_BATCH = 2


def build_lstm_ids_scada():
    return LSTMForecastDetector(
        n_features=_N_FEATURES,
        units=_UNITS,
        history=_HISTORY,
        layers=_LAYERS,
        dropout=0.5,
    )


def example_input_lstm_ids_scada():
    return torch.randn(_BATCH, _HISTORY, _N_FEATURES)


MENAGERIE_ENTRIES = [
    (
        "LSTM-IDS-SCADA",
        build_lstm_ids_scada,
        example_input_lstm_ids_scada,
        2020,
        "PORT",
    ),
]
