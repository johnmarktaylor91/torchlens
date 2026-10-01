# SOURCE: vendored from locuslab/TCN @ master
# (https://github.com/locuslab/TCN, TCN/tcn.py: Chomp1d, TemporalBlock, TemporalConvNet;
# TCN/adding_problem/model.py: sequence-to-scalar regression head pattern, "self.linear(y1[:, :, -1])")
#
# "1D-TCN for NIR aflatoxin detection" (candidate queue name) has no dedicated public repo, but its
# own triage note describes it as "a standard TCN block + 1D spectral input" -- i.e. the contribution
# is the *application* (per-pixel near-infrared hyperspectral spectral sequence -> aflatoxin B1
# detection), not a novel architecture. The underlying block is exactly the canonical dilated-causal
# 1D temporal convolutional network from Bai, Kolter & Koltun, "An Empirical Evaluation of Generic
# Convolutional and Recurrent Networks for Sequence Modeling" (2018), whose reference implementation
# is locuslab/TCN. Per the source ladder this is RUNG 2: the real repo's model-definition file is
# vendored verbatim (only the head is swapped for the last-timestep regression/classification pattern
# the SAME repo already uses for its own sequence-to-scalar tasks, e.g. TCN/adding_problem/model.py),
# not a from-scratch reimplementation of "a TCN" from a paper summary.
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils import weight_norm

MENAGERIE_ZOO = "vendored-pytorch"


# --- verbatim from locuslab/TCN @ master, TCN/tcn.py ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- application head: per-pixel NIR spectral sequence -> AFB1 (aflatoxin B1) contamination class,
# following the SAME repo's own last-timestep sequence-to-scalar head pattern
# (TCN/adding_problem/model.py: "self.linear(y1[:, :, -1])") ---
class AflatoxinTCN(nn.Module):
    """1D-TCN aflatoxin detector: dilated causal TCN over a per-pixel NIR spectral curve."""

    def __init__(
        self,
        num_bands: int = 1,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        num_channels = num_channels or [16, 16, 32]
        self.tcn = TemporalConvNet(
            num_bands, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        self.classifier.weight.data.normal_(0, 0.01)

    def forward(self, spectrum: Tensor) -> Tensor:
        """spectrum: (batch, num_bands, num_wavelengths) per-pixel NIR spectral sequence."""
        features = self.tcn(spectrum)
        return self.classifier(features[:, :, -1])


def build_aflatoxin_tcn() -> nn.Module:
    model = AflatoxinTCN(
        num_bands=1, num_channels=[16, 16, 32], kernel_size=3, dropout=0.1, num_classes=2
    )
    model.eval()
    return model


def example_input_aflatoxin_tcn() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 1, 64)  # (batch, 1 per-pixel channel, 64 NIR spectral bands)


MENAGERIE_ENTRIES = [
    (
        "1D-TCN for NIR aflatoxin detection",
        "build_aflatoxin_tcn",
        "example_input_aflatoxin_tcn",
        2022,
        MENAGERIE_ZOO,
    ),
]
