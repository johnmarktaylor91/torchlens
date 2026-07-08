# SOURCE: vendored from att-ar/pytorch_colab @ main
# https://github.com/att-ar/pytorch_colab/blob/main/battery_lstm.py
# https://github.com/att-ar/pytorch_colab/blob/main/global_dataclass.py
# `LSTMNetwork` (battery state-of-charge estimator from current/voltage/SoC
# time series -- LSTM followed by a BatchNorm1d/ReLU/Dropout MLP stack with a
# final Sigmoid) is transcribed VERBATIM from the real repo, only rewriting
# the `G` config dataclass import into inline defaults matching
# `global_dataclass.py` (num_features=3, lstm_nodes=256) and dropping the
# CPU/GPU `G.device` indirection used only in the dataset class (not part of
# the model architecture).
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

NUM_FEATURES = 3  # current, voltage, soc -- matches global_dataclass.G.num_features
LSTM_NODES = 256  # matches global_dataclass.G.lstm_nodes


class LSTMNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(NUM_FEATURES, LSTM_NODES, 1, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.uniform_(param, a=0.001, b=0.09)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(
                    param, nonlinearity="relu"
                )  # PyTorch equivalent to He Normalization
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.linear_stack = nn.Sequential(
            nn.Linear(LSTM_NODES, LSTM_NODES),  # shape == (G.batch_size, G.lstm_nodes)
            nn.BatchNorm1d(LSTM_NODES, momentum=0.92),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(LSTM_NODES, LSTM_NODES),  # shape == (G.batch_size, G.lstm_nodes)
            nn.BatchNorm1d(LSTM_NODES, momentum=0.92),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(LSTM_NODES, 1),  # shape == (G.batch_size, 1)
            Sigmoid(),
        )
        for i in range(0, len(self.linear_stack), 2):
            for name, param in self.linear_stack[i].named_parameters():
                if "bias" in name:
                    nn.init.uniform_(param, a=(0.1 / (i / 2 + 1)), b=(0.9 / (i / 2 + 1)))
                elif "weight" in name:
                    nn.init.kaiming_normal_(param, nonlinearity="relu")

    def forward(self, x):  # assert(x.shape == (G.batch_size, G.window_size, G.num_features))
        # lstm
        x_out, (h_n_lstm, c_n) = self.lstm(
            x
        )  # assert(h_n_lstm.shape == (1, G.batch_size, G.lstm_nodes))
        # Dense Layers
        # send the final lstm layer's hidden state values to the Dense Layers
        out = self.linear_stack(h_n_lstm.squeeze())
        return out  # (G.batch_size, 1)


MENAGERIE_ZOO = "vendored-pytorch"


def build_lstm_soc():
    torch.manual_seed(0)
    model = LSTMNetwork()
    model.eval()
    return model


def example_input_lstm_soc():
    torch.manual_seed(0)
    # (batch_size, window_size, num_features); batch_size=2 avoids the
    # h_n_lstm.squeeze() call collapsing the batch dim to a 1D vector for
    # batch_size==1, matching real training-time usage (G.batch_size=16).
    return torch.randn(2, 16, NUM_FEATURES)


MENAGERIE_ENTRIES = [
    (
        "LSTM-SOC (battery state of charge)",
        "build_lstm_soc",
        "example_input_lstm_soc",
        2022,
        MENAGERIE_ZOO,
    ),
]
