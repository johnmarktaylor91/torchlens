# SOURCE: vendored from
# https://github.com/dspanah/Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch
# @ master (notebooks/1.0-dsp-DeepConvLSTM.ipynb, cell defining `HARModel`)
#
# DeepConvLSTM (Ordonez & Roggen 2016, Sensors 16(1):115) for wearable-sensor
# human activity recognition: four stacked 1D-conv feature-extraction layers
# over a sliding time window of multichannel sensor data, followed by two
# stacked LSTM layers over the resulting feature sequence, then a linear
# classifier head applied to the final LSTM timestep. This PyTorch port
# (the repo `dspanah/...DeepConvLSTM-Pytorch`) implements the architecture
# from the original Ordonez & Roggen paper; `HARModel.__init__`/`forward`/
# `init_hidden` are copied verbatim below (only the free-standing notebook
# globals NB_SENSOR_CHANNELS / SLIDING_WINDOW_LENGTH / train_on_gpu, and the
# `metrics.f1_score`-based training loop, are outside forward() and are not
# needed for tracing the network itself).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# Real notebook constants (OPPORTUNITY dataset: 113 sensor channels, 24-frame
# sliding window). The `x.view(8, -1, self.n_filters)` reshape inside
# forward() is HARD-CODED in the original notebook to assume
# SLIDING_WINDOW_LENGTH=24 with filter_size=5 (four Conv1d layers each with
# no padding shrink the time axis by filter_size-1=4, so 24 - 4*4 = 8) --
# these constants are architecturally load-bearing and are kept at their
# real values.
NB_SENSOR_CHANNELS = 113
SLIDING_WINDOW_LENGTH = 24


class HARModel(nn.Module):
    """Verbatim from the notebook's model-definition cell."""

    def __init__(
        self, n_hidden=128, n_layers=1, n_filters=64, n_classes=18, filter_size=5, drop_prob=0.5
    ):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size

        self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)

        self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden, batch_size):
        x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(8, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)

        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)

        out = x.view(batch_size, -1, self.n_classes)[:, -1, :]

        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state (verbatim, CPU-only branch)."""
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
        )
        return hidden


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). `forward()` needs
# THREE inputs (x, hidden, batch_size) so this ships as a module (not a
# recipe). We wrap HARModel so its real forward is called with a real
# initial hidden state built by the real `init_hidden`, matching how the
# notebook's `train()` loop drives the model.
# ---------------------------------------------------------------------------
class HARModelWrapped(nn.Module):
    def __init__(self, har_model, batch_size):
        super().__init__()
        self.har_model = har_model
        self.batch_size = batch_size

    def forward(self, x):
        hidden = self.har_model.init_hidden(self.batch_size)
        out, _ = self.har_model(x, hidden, self.batch_size)
        return out


_BATCH = 4


def build_deepconvlstm_harmodel():
    torch.manual_seed(0)
    har = HARModel(
        n_hidden=32, n_layers=1, n_filters=16, n_classes=18, filter_size=5, drop_prob=0.5
    )
    har.eval()
    model = HARModelWrapped(har, _BATCH)
    model.eval()
    return model


def example_input_deepconvlstm_harmodel():
    torch.manual_seed(0)
    return torch.randn(_BATCH, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)


MENAGERIE_ENTRIES = [
    (
        "DeepConvLSTM-HARModel",
        "build_deepconvlstm_harmodel",
        "example_input_deepconvlstm_harmodel",
        2016,
        MENAGERIE_ZOO,
    ),
]
