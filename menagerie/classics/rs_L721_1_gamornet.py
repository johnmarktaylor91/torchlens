# FAITHFUL PORT of aritraghsh09/GaMorNet @ master (original framework: Keras/TF)
# File: gamornet/keras_module.py (gamornet_build_model_keras, LocalResponseNormalization)
#
# GaMorNet (Ghosh et al. 2020, ApJ) -- an AlexNet-style CNN for automated galaxy
# morphology classification (disk / indeterminate / spheroid) from SDSS/CANDELS
# postage-stamp images. Faithfully transcribed layer-for-layer from the real Keras
# `gamornet_build_model_keras()`: 5 conv blocks (96/256/384/384/256 filters, kernel
# sizes 11/5/3/3/3, stride 4 on the first conv) with ReLU, 3x3/stride-2 max pooling
# and Local Response Normalization (LRN) after blocks 1, 2, and 5, followed by
# Flatten -> Dense(4096, tanh) -> Dropout(0.5) -> Dense(4096, tanh) -> Dropout(0.5)
# -> Dense(3, softmax). LRN is transcribed from the repo's own from-scratch Keras
# `LocalResponseNormalization` layer (itself credited in-repo to Chollet's "Deep
# Learning with Python") using torch's native `nn.LocalResponseNorm(n=5, alpha=1e-4,
# beta=0.75, k=1.0)`, which implements the identical cross-channel local response
# normalization formula. Kernel initializers (VarianceScaling fan_in/uniform,
# TruncatedNormal) are dropped in favor of torch defaults -- weight init, not
# architecture. Dropped: the SDSS/CANDELS weight-download & transfer-learning
# helpers (data/training utilities, not the network).
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class GaMorNet(nn.Module):
    """AlexNet-style CNN with LRN, ported from gamornet_build_model_keras()."""

    def __init__(self, in_channels: int = 1, input_hw: int = 167, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0),
        )
        with torch.no_grad():
            probe = self.features(torch.zeros(1, in_channels, input_hw, input_hw))
        flat_dim = probe.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 4096),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_gamornet():
    torch.manual_seed(0)
    # Tiny stand-in for the real 167x167 SDSS input size, keeps the trace fast.
    return GaMorNet(in_channels=1, input_hw=32, num_classes=3)


def example_input_gamornet():
    torch.manual_seed(0)
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("GaMorNet", "build_gamornet", "example_input_gamornet", 2020, "SOURCE_AVAILABLE"),
]
