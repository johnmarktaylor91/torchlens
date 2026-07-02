# FAITHFUL PORT of https://github.com/khundman/telemanom @ master
# (telemanom/modeling.py Model.train_new, original framework: Keras/TensorFlow)
#
# Telemanom (NASA JPL LSTM-based unsupervised anomaly detection for
# multivariate spacecraft telemetry, Hundman et al. KDD 2018) -- the real
# repo's `Model.train_new` builds this exact Keras `Sequential` stack (see
# telemanom/modeling.py, config defaults in config.yaml: layers=[80, 80],
# dropout=0.3, n_predictions=10):
#     LSTM(80, return_sequences=True) -> Dropout(0.3) ->
#     LSTM(80, return_sequences=False) -> Dropout(0.3) ->
#     Dense(n_predictions) -> Activation('linear')
# The repo ships only Keras/TF (`from keras.models import Sequential`,
# `from keras.layers.recurrent import LSTM`) with no PyTorch implementation
# anywhere in the project, and TensorFlow/Keras is not installed in this
# environment (protobuf/MessageFactory import failure), so the real code
# cannot be run or vendored directly. This is a faithful 1:1 layer-for-layer
# torch transcription of the real `train_new` architecture above -- same
# two stacked LSTM widths, same dropout rate and placement, same final
# Dense+linear-activation regression head predicting `n_predictions` future
# telemetry steps from an input window. Training-loop code (EarlyStopping,
# History callbacks, model.fit/compile) is not architecture and is dropped.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class TelemanomLSTM(nn.Module):
    def __init__(self, n_features, layers=(80, 80), dropout=0.3, n_predictions=10):
        super().__init__()
        hidden1, hidden2 = layers
        # Keras LSTM(80, return_sequences=True): a stacked-LSTM layer whose
        # full output sequence feeds the next LSTM layer.
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        # Keras LSTM(80, return_sequences=False): only the last timestep's
        # hidden state is kept, matching Keras's default return_sequences=False.
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        # Dense(n_predictions) + Activation('linear') == a plain Linear layer.
        self.dense = nn.Linear(hidden2, n_predictions)

    def forward(self, x):
        # x: (batch, timesteps, n_features)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, (hn, cn) = self.lstm2(x)
        x = x[:, -1, :]  # return_sequences=False: keep only the last timestep
        x = self.dropout2(x)
        x = self.dense(x)
        return x


def build_telemanom():
    # Real repo defaults: layers=[80, 80], dropout=0.3, n_predictions=10.
    # n_features=8 is a stand-in small channel count (the real repo's
    # per-channel telemetry width varies by SMAP/MSL channel; the repo does
    # not hardcode a single fixed value).
    return TelemanomLSTM(n_features=8, layers=(80, 80), dropout=0.3, n_predictions=10)


def example_input_telemanom():
    # Real repo default l_s (input window length) is 250; a smaller window
    # of 20 timesteps is used here to keep the trace lightweight while
    # exercising the identical layer stack.
    return torch.randn(3, 20, 8)


MENAGERIE_ENTRIES = [
    (
        "Telemanom",
        build_telemanom,
        example_input_telemanom,
        2018,
        MENAGERIE_ZOO,
    ),
]
