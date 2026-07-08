# SOURCE: vendored from https://github.com/datacubeR/DeepAnt @ master (deepant.py)
# DeepAnT: predict-then-detect time-series anomaly detector (Munir et al., IEEE
# Access 2019, "DeepAnT: A Deep Learning Approach for Unsupervised Anomaly
# Detection in Time Series"). Two 1D-conv blocks predict the next `p_w` values
# of a univariate window; large prediction error flags an anomaly.
#
# Vendored real repo code (the `DeepAnt` nn.Module from deepant.py). Only the
# non-architectural PyTorch Lightning training wrapper (AnomalyDetector),
# Dataset/DataModule scaffolding, and sklearn preprocessing were dropped --
# none of that is part of the architecture. No layer, kernel size, channel
# count, or dataflow inside DeepAnt itself was changed.

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class DeepAnt(nn.Module):
    """Two 1D-conv blocks (conv -> relu -> maxpool) feeding a dense predictor
    head that forecasts the next `p_w` timesteps from a length-`seq_len`
    univariate window."""

    def __init__(self, seq_len, p_w):
        super().__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.denseblock = nn.Sequential(
            nn.Linear(32, 40),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.out = nn.Linear(40, p_w)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x


def build_deepant():
    # seq_len=10 is the smallest window that satisfies the real repo's fixed
    # Linear(32, 40) bottleneck (two conv/pool halvings must collapse the
    # sequence axis to length 1); p_w=1 predicts one step ahead.
    return DeepAnt(seq_len=10, p_w=1)


def example_input_deepant():
    # (batch, channels=1, seq_len=10) univariate window, matching
    # TrafficDataset's `.permute(1, 0)` layout in the real repo.
    return torch.randn(4, 1, 10)


MENAGERIE_ENTRIES = [
    (
        "DeepAnT",
        build_deepant,
        example_input_deepant,
        2019,
        MENAGERIE_ZOO,
    ),
]
