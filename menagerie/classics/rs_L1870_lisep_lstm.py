# FAITHFUL PORT of Raahul46/Early-sepsis-detection @ master (original framework: Keras/TensorFlow)
# https://github.com/Raahul46/Early-sepsis-detection / sepsis.ipynb
#
# The linked community re-implementation of "LiSep LSTM" (Fagerstrom et al.,
# Scientific Reports 2019, "LiSep LSTM: A Machine Learning Algorithm for Early
# Detection of Septic Shock", https://www.nature.com/articles/s41598-019-51219-4)
# is Keras/TF-only (model.h5 + sepsis.ipynb, no PyTorch source), so it cannot be
# vendored directly. The exact architecture cell from that notebook is:
#
#   model = Sequential()
#   model.add(LSTM(128, input_shape=(30, 11), return_sequences=True))
#   model.add(LSTM(128))
#   model.add(Dense(64, activation='relu'))
#   model.add(Dense(2, activation='softmax'))
#
# i.e. a stacked 2-layer LSTM (first layer returns the full sequence, second
# layer returns only the final hidden state) feeding a Dense(64, ReLU) ->
# Dense(2, softmax) classification head, over 30-timestep windows of 11
# reduced MIMIC-III vital-sign/lab features (matching the paper's septic-shock
# early-warning task, itself replicating the TREWScore feature set from
# Henry et al. 2015). This module transcribes that Keras graph layer-for-layer
# into torch: nn.LSTM(11, 128, num_layers=1) stacked twice (matching Keras'
# stateless per-layer LSTM stacking, not merging into one multi-layer LSTM
# module, to keep the return_sequences=True / False layer-1-vs-layer-2
# distinction explicit and faithful) -> Linear(128, 64) + ReLU ->
# Linear(64, 2) + softmax.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class LiSepLSTM(nn.Module):
    """Faithful torch port of the Keras LiSep LSTM graph:
    LSTM(128, return_sequences=True) -> LSTM(128) -> Dense(64, relu) -> Dense(2, softmax).
    """

    def __init__(self, n_features: int = 11, hidden_size: int = 128, seq_len: int = 30):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Keras LSTM(128, return_sequences=True): full hidden-state sequence out.
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        # Keras LSTM(128) (return_sequences default False): only final hidden state out.
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        self.dense1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_features]
        seq_out, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(seq_out)
        last_hidden = h_n[-1]  # [batch, hidden_size], final layer's hidden state
        out = self.relu(self.dense1(last_hidden))
        out = self.dense2(out)
        out = self.softmax(out)
        return out


def build_lisep_lstm():
    model = LiSepLSTM(n_features=11, hidden_size=128, seq_len=30)
    model.eval()
    return model


def example_input_lisep_lstm():
    return torch.randn(2, 30, 11)


MENAGERIE_ENTRIES = [
    ("LiSep LSTM", build_lisep_lstm, example_input_lisep_lstm, 2019, "ported-pytorch"),
]
