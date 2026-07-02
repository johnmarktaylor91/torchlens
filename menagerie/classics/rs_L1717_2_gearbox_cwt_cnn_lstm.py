# SOURCE: vendored from yriyazi/Enhanced-Gearbox-Fault-Diagnosis-with-Fusion-LSTM-CNN-Network-ISAV_2023 @ main
#
# Vendors the real nn.Module classes from Model/cnn_net.py (InceptionBlock), Model/lstm_net.py
# (LSTMModel), and Model/structure.py (Structure_CNN_RNN fusion head), with only import paths
# and config-loading (originally driven by Utils/configuration.py + config.yaml) inlined as
# module-level constants so the file is self-contained. Architecture is untouched: a multi-branch
# Inception-style 2D CNN over Continuous-Wavelet-Transform (CWT) coefficients of the raw vibration
# signal, fused via concatenation with an LSTM branch over the raw time series, into a small MLP
# classifier. NOTE: despite the catalog queue's "WPD" (wavelet packet decomposition) paraphrase,
# the actual repo code uses `pywt.cwt` (continuous wavelet transform), not WPD -- the real source
# is authoritative over the queue's summary.
#
# Repo: https://github.com/yriyazi/Enhanced-Gearbox-Fault-Diagnosis-with-Fusion-LSTM-CNN-Network-ISAV_2023 @ main
# Files: Model/cnn_net.py, Model/lstm_net.py, Model/structure.py, Utils/utils.py (device),
#        Utils/configuration.py + config.yaml (hyperparameter constants, inlined below).
#
# NOTE (harness accommodation, not an architecture change): the original `forward(x: np.array)`
# takes a raw numpy 1-D signal. TorchLens's capture entry point requires a concrete torch.Tensor
# module input, so `Structure_CNN_RNN.forward` below accepts a torch.Tensor and converts to numpy
# (`.detach().cpu().numpy()`) before feeding `pywt.cwt`, exactly mirroring what the caller of the
# original repo function would have passed in (a 1-D array of raw samples). No layer, mechanism,
# or computation differs from the source.

import numpy as np
import pywt
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

# ---- config.yaml values (Utils/configuration.py originally sourced these from YAML) ----
Device = torch.device("cpu")
input_horizon = 200
LSTM_outFeature = 60
LSTM_NumLayer = 1
LSTM_hidden_size = 50
CNN_inChannel = 1
CNN_outChannel = 10
CNN_outFeature = 80
Scales = 17
wavelet_lis = ["mexh", "morl", "gaus1", "cmor1-1.0"]
wavelet = wavelet_lis[1]
Coefficient_Real = True


# ---- Model/cnn_net.py ----
class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channel: int):
        """
        Initialize the InceptionBlock with various branches for different convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channel (int): Number of output channels.
        """
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel

        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1), nn.ReLU(inplace=True)
        )

        # 1x1 conv followed by 3x3 conv branch
        self.conv3x3 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv followed by 5x5 conv branch
        self.conv5x5 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv followed by 9x9 conv branch
        self.conv9x9 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=9, padding="same"),
            nn.ReLU(inplace=True),
        )
        # 3x3 pooling followed by 1x1 conv branch
        self.conv1x1_pool = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.out_channel * 5, CNN_outFeature, kernel_size=(4, 16), stride=2, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_outFeature, CNN_outFeature, kernel_size=(4, 16), stride=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def GAP(self, x: torch.Tensor) -> torch.Tensor:
        """Global Average Pooling (GAP) operation."""
        return torch.mean(x, dim=[2, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out9x9 = self.conv9x9(x)
        out1x1_pool = self.conv1x1_pool(x)

        out = torch.cat([out1x1, out3x3, out5x5, out9x9, out1x1_pool], dim=1)

        out = self.GAP(self.head(out))
        return out


# ---- Model/lstm_net.py ----
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_horizon: int,
        hidden_size: int = LSTM_hidden_size,
        num_layers: int = LSTM_NumLayer,
        output_size: int = LSTM_outFeature,
        DropOut: float = 0,
    ) -> None:
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_horizon, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_state, _) = self.lstm(x)

        if len(out.shape) == 2:
            out = self.fc(h_state)
        elif len(out.shape) == 3:
            out = self.fc(out[:, -1, :])

        return out


# ---- Model/structure.py ----
class _CWT:
    def __init__(self) -> None:
        self.scales = np.arange(1, Scales)

    def forward(self, x):
        coefficients, _ = pywt.cwt(x, self.scales, wavelet)
        if Coefficient_Real:
            coefficients = np.abs(coefficients)

        return coefficients


class Structure_CNN_RNN(nn.Module):
    def __init__(self):
        super(Structure_CNN_RNN, self).__init__()

        self.LSTM = LSTMModel(input_horizon).to(device=Device)
        self.CNN = InceptionBlock(CNN_inChannel, CNN_outChannel).to(device=Device)

        self.CWT = _CWT()
        _out = LSTM_outFeature + CNN_outFeature
        self.Classifier = nn.Sequential(
            nn.Linear(_out, _out // 5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(_out // 5, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        coefficients = self.CWT.forward(x_np)

        _L = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device=Device)
        _L_out = self.LSTM(_L)

        _C = (
            torch.tensor(coefficients, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=Device)
        )
        _C_out = self.CNN(_C)

        out = torch.cat([_C_out, _L_out], dim=1)
        out = self.Classifier(out)

        return out


def build_gearbox_cwt_cnn_lstm():
    return Structure_CNN_RNN()


def example_input_gearbox_cwt_cnn_lstm():
    # Structure_CNN_RNN.forward performs the CWT internally on the raw 1-D signal
    # (matching the original repo's calling convention), then wraps it in torch tensors.
    return torch.randn(input_horizon)


MENAGERIE_ENTRIES = [
    (
        "Gearbox Vibration CWT-CNN-LSTM Fusion",
        build_gearbox_cwt_cnn_lstm,
        example_input_gearbox_cwt_cnn_lstm,
        2023,
        "SOURCE_AVAILABLE",
    ),
]
