# FAITHFUL PORT of interseismic/generalized-phase-detection @ master (original framework: Keras/TF)
# File: model_pol.json (serialized Keras functional-model config loaded by gpd_predict.py
#        via `model_from_json`)
#
# GPD -- Generalized Phase Detection (Ross et al. 2018, BSSA, "Generalized Seismic Phase
# Detection with Deep Learning") -- a 1D CNN that classifies 400-sample, 3-component
# (E/N/Z) seismic waveform windows into P-wave / S-wave / noise. The real architecture
# ships only as a Keras model_from_json blob (`model_pol.json`, no from-scratch Python
# builder in the repo); transcribed here layer-for-layer from that JSON: 4 Conv1D blocks
# (32/64/128/256 filters, kernel sizes 21/15/11/9, "same" padding) each followed by
# BatchNorm1d, ReLU, and 2x/stride-2 max pooling, then Flatten -> Dense(200) -> BatchNorm
# -> ReLU -> Dense(200) -> BatchNorm -> ReLU -> Dense(3) -> softmax. The JSON's outer
# `lambda_1/2/3` + triple-replicated `sequential_1` + final `Concatenate` layers are Keras's
# `multi_gpu_model()` batch-splitting wrapper (an infra artifact of how this checkpoint's
# graph was serialized for 3-GPU training/inference), not part of the semantic network --
# the true model is the single `sequential_1` CNN branch applied once per window, which is
# what is ported here. Dropped: ObsPy waveform I/O, sliding-window inference loop, and
# multi-GPU replication wrapper -- data/serving utilities, not the network.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class GPD(nn.Module):
    """1D CNN seismic phase-picker, ported from model_pol.json's `sequential_1` branch."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=21, stride=1, padding=10),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            probe = self.features(torch.zeros(1, in_channels, 400))
        flat_dim = probe.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_gpd():
    torch.manual_seed(0)
    return GPD(in_channels=3, num_classes=3)


def example_input_gpd():
    torch.manual_seed(0)
    # real model uses fixed 400-sample, 3-component (E/N/Z) windows
    return torch.randn(2, 3, 400)


MENAGERIE_ENTRIES = [
    ("GeneralizedPhaseDetection", "build_gpd", "example_input_gpd", 2018, "SOURCE_AVAILABLE"),
]
