# FAITHFUL PORT of skolpin/swat-anomaly-detection @ main (original framework: TensorFlow/Keras)
#
# Source: rnn_training.py, function create_model() (Master's-thesis research on LSTM-based
# anomaly detection in the SWaT (Secure Water Treatment) industrial-control testbed). The
# original Keras graph is a stateful-LSTM one-step-ahead forecaster used as a reconstruction/
# forecast-error anomaly detector:
#
#   model = Sequential()
#   model.add(LSTM(neurons, batch_input_shape=(batch_size, 1, n_features),
#                   stateful=True, return_sequences=False))
#   model.add(Dense(5, activation='tanh'))
#   model.add(Dense(out_dim, activation='tanh'))
#
# The model consumes a length-1 "sequence" of the current multivariate sensor sample
# (LSTM input shape (batch, 1, n_features) -- the recurrence carries state *across calls*
# via Keras `stateful=True`, not across timesteps within one call) and predicts the next
# sample (out_dim = number of signals being modeled: 1 for the single-signal LIT101/DPIT301/
# LIT301 notebooks, 3 for the Multisignal notebook). This port transcribes the exact layer
# stack -- one `nn.LSTM(neurons)` cell, `nn.Linear(neurons, 5)` + Tanh, `nn.Linear(5, out_dim)`
# + Tanh -- as an ordinary (non-stateful) torch module: the Keras `stateful=True` flag only
# changes *how hidden state is carried across separate .fit()/.predict() calls* during
# training/inference in the original code; it is not an architectural layer and has no
# effect on a single forward pass, which is what TorchLens captures here. No layer, width,
# or nonlinearity was changed from the real repo's create_model().
#
# Repo: https://github.com/skolpin/swat-anomaly-detection @ main, rnn_training.py

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class LSTMSwatForecaster(nn.Module):
    """Faithful port of create_model() (rnn_training.py): stateful-LSTM one-step
    forecaster used as a reconstruction-error anomaly detector on SWaT sensor
    signals (LIT101 / DPIT301 / LIT301 / multi-signal variants).
    """

    def __init__(self, n_features: int = 1, neurons: int = 10, out_dim: int = 1):
        super().__init__()
        self.n_features = n_features
        self.out_dim = out_dim
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=neurons, batch_first=True)
        self.dense_1 = nn.Linear(neurons, 5)
        self.act_1 = nn.Tanh()
        self.dense_2 = nn.Linear(5, out_dim)
        self.act_2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_features) -- single-timestep window, matching the
        # original `X.reshape(X.shape[0], 1, X.shape[1])` call site.
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        h = self.act_1(self.dense_1(last_step))
        return self.act_2(self.dense_2(h))


def build_lstm_swat_single():
    # LIT101 / DPIT301 / LIT301 single-signal notebooks: 1 input feature, 1 output.
    return LSTMSwatForecaster(n_features=1, neurons=10, out_dim=1)


def example_input_lstm_swat_single():
    return torch.randn(4, 1, 1)


def build_lstm_swat_multisignal():
    # Multisignal.ipynb: multivariate input (LIT101, DPIT301, LIT301 together),
    # multivariate one-step-ahead output over the same 3 channels.
    return LSTMSwatForecaster(n_features=3, neurons=10, out_dim=3)


def example_input_lstm_swat_multisignal():
    return torch.randn(4, 1, 3)


MENAGERIE_ENTRIES = [
    (
        "LSTM Autoencoder FDD for SWaT",
        build_lstm_swat_single,
        example_input_lstm_swat_single,
        2021,
        "PORT",
    ),
    (
        "LSTM SWaT Cyber-Physical Multisignal",
        build_lstm_swat_multisignal,
        example_input_lstm_swat_multisignal,
        2021,
        "PORT",
    ),
]
