# SOURCE: vendored from https://github.com/harvard-edge/cs249r_book @ main
# (mlperf-edu/reference/tiny/dscnn_kws.py)
#
# DS-CNN for keyword spotting: the MLPerf Tiny (MLCommons) reference
# architecture for the keyword-spotting benchmark task (Zhang, Suda, Lai,
# Chandra, "Hello Edge: Keyword Spotting on Microcontrollers", 2017),
# reproduced as a runnable PyTorch reference model in Harvard's cs249r_book
# "TinyML/MLPerf-EDU" teaching repository (the official mlcommons/tiny KWS
# reference is Keras/TF-only; this is the real PyTorch `DSCNN`/`DSCNNBlock`
# `nn.Module` code from that repo, not a from-scratch reimplementation).
# Only the dataset/dataloader/`__main__` demo code was trimmed for menagerie
# staging; the `DSCNNBlock` and `DSCNN` classes are copied verbatim -- a
# depthwise-separable-convolution stem-and-stack feeding global average pool
# and a linear classifier over mel-spectrogram input.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class DSCNNBlock(nn.Module):
    """Depthwise-Separable Convolution Block.

    Splits the standard convolution into:
    1. Depthwise: one spatial filter per input channel (groups=in_channels)
    2. Pointwise: 1x1 convolution to mix channels
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """
    DS-CNN for keyword spotting (MLPerf Tiny reference architecture).

    Input: Mel spectrogram of shape (B, 1, n_mels, time_steps)
    Output: (B, num_classes) logits

    The model is deliberately small (~60K parameters) to fit on a
    microcontroller with <256KB SRAM.
    """

    def __init__(self, num_classes=12, n_mels=40):
        super().__init__()

        # Initial convolution: maps 1-channel spectrogram to 64 filters
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 4 DS-CNN blocks (same channel dimension for simplicity)
        self.ds_blocks = nn.Sequential(
            DSCNNBlock(64, 64),
            DSCNNBlock(64, 64),
            DSCNNBlock(64, 48),
            DSCNNBlock(48, 48),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x, targets=None):
        """
        Forward pass compatible with the auto_trainer interface.

        Args:
            x: (B, 1, n_mels, time_steps) mel spectrogram
            targets: (B,) class labels for loss computation

        Returns:
            logits: (B, num_classes)
            loss: scalar if targets provided
        """
        x = self.conv_init(x)
        x = self.ds_blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        logits = self.fc(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_NUM_CLASSES = 12
_N_MELS = 40
_TIME_STEPS = 101  # ~1s of audio at the repo's hop_length=160, n_fft=480 mel transform


def build_dscnn_kws():
    torch.manual_seed(0)
    model = DSCNN(num_classes=_NUM_CLASSES, n_mels=_N_MELS)
    model.eval()
    return model


def example_input_dscnn_kws():
    torch.manual_seed(0)
    return torch.randn(1, 1, _N_MELS, _TIME_STEPS)


MENAGERIE_ENTRIES = [
    (
        "DS-CNN (MLPerf Tiny Keyword Spotting)",
        "build_dscnn_kws",
        "example_input_dscnn_kws",
        2017,
        MENAGERIE_ZOO,
    ),
]
