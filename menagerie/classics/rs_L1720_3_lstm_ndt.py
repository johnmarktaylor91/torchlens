# FAITHFUL PORT of khundman/telemanom @ master (original framework: Keras/TensorFlow)
#
# Ported from: telemanom/modeling.py, class Model.train_new()
#
# LSTM-NDT (KDD 2018, Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
# Nonparametric Dynamic Error Thresholding"): the official NASA SMAP/MSL anomaly
# detector. The forecasting network itself (the piece TorchLens traces) is the exact
# stacked-LSTM regressor built in `Model.train_new`:
#
#     model = Sequential()
#     model.add(LSTM(layers[0], input_shape=(None, n_features), return_sequences=True))
#     model.add(Dropout(dropout))
#     model.add(LSTM(layers[1], return_sequences=False))
#     model.add(Dropout(dropout))
#     model.add(Dense(n_predictions))
#     model.add(Activation('linear'))
#
# i.e. two stacked LSTM layers (the first returns the full sequence, the second only the
# final hidden state) each followed by dropout, then a linear projection to
# `n_predictions` forecasted values (linear activation = identity). The nonparametric
# dynamic error thresholding (NDT) itself is a post-hoc numpy statistic over the
# residuals, not a network layer, so it is outside the traced module; this port keeps
# only the differentiable forecasting network, transcribed 1:1 layer-for-layer from the
# real repo code (default config.yaml values: layers=[80, 80], dropout=0.3).

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class LSTM_NDT(nn.Module):
    """Faithful torch port of telemanom.modeling.Model's Keras Sequential network:
    LSTM(layers[0], return_sequences=True) -> Dropout
      -> LSTM(layers[1], return_sequences=False) -> Dropout
      -> Dense(n_predictions) -> Activation('linear')."""

    def __init__(self, n_features, layers=(80, 80), n_predictions=10, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=layers[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=layers[0], hidden_size=layers[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.dense = nn.Linear(layers[1], n_predictions)
        # Activation('linear') in Keras is the identity function.

    def forward(self, x):
        out, _ = self.lstm1(x)  # return_sequences=True
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # return_sequences=False
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.dense(out)
        return out


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_N_FEATURES = 3  # SMAP/MSL channels are typically 1 telemetry + command-encoded dims
_HISTORY_LEN = 20  # config.yaml default l_s=250 shrunk for tracing
_LAYERS = (16, 16)  # config.yaml default (80, 80) shrunk for tracing
_N_PREDICTIONS = 10
_BATCH = 2


def build_lstm_ndt():
    return LSTM_NDT(
        n_features=_N_FEATURES,
        layers=_LAYERS,
        n_predictions=_N_PREDICTIONS,
        dropout=0.3,
    )


def example_input_lstm_ndt():
    return torch.randn(_BATCH, _HISTORY_LEN, _N_FEATURES)


MENAGERIE_ENTRIES = [
    (
        "LSTM-NDT",
        build_lstm_ndt,
        example_input_lstm_ndt,
        2018,
        "PORT",
    ),
]
