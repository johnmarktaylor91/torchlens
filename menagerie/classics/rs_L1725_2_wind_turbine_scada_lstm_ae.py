# SOURCE: vendored from What-a-mess/Wind-Turbine-SCADA-Anomaly-Detection @ 207575337e5edf5a013319e840c44521db1f755c
#
# `LSTMAutoEncoder` (autoencoder/model.py) is the LSTM autoencoder used by this repo to
# detect anomalies in wind-turbine SCADA sensor streams (Penmanshiel Wind Farm dataset):
# an MLP feature-compression stack feeding a single-step `nn.LSTM` bottleneck (encoder),
# mirrored by an `nn.LSTM` expansion feeding an MLP decompression stack (decoder), trained
# to reconstruct its own input (reconstruction error = anomaly score). Copied verbatim
# from the real repo's `autoencoder/model.py`; only the module-level
# `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` global and the
# per-submodule `.to(device)` calls are kept as in the original (they resolve to "cpu" in
# this environment, matching the original code's own device-selection logic -- not a
# behavioral change).
#
# Repo: https://github.com/What-a-mess/Wind-Turbine-SCADA-Anomaly-Detection @ master
# File vendored: autoencoder/model.py (LSTMAutoEncoder + its GetRnnOutput helper)

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GetRnnOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size: int, inner_size=20):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LSTM(32, inner_size),
            GetRnnOutput(),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.LSTM(inner_size, 32),
            GetRnnOutput(),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
        ).to(device)

    def forward(self, x):
        encode_res = self.encoder(x)
        decode_res = self.decoder(encode_res)

        return decode_res


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_INPUT_SIZE = 16
_SEQ_LEN = 12
_BATCH = 4


def build_wind_turbine_scada_lstm_ae():
    return LSTMAutoEncoder(input_size=_INPUT_SIZE, inner_size=20)


def example_input_wind_turbine_scada_lstm_ae():
    return torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE)


MENAGERIE_ENTRIES = [
    (
        "Wind Turbine SCADA LSTM-AE",
        build_wind_turbine_scada_lstm_ae,
        example_input_wind_turbine_scada_lstm_ae,
        2022,
        "SOURCE_AVAILABLE",
    ),
]
