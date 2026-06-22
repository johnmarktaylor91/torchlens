"""LipNet: end-to-end sentence-level lipreading with STCNN + BiGRU + CTC head.

Assael et al. (2016), "LipNet: End-to-End Sentence-level Lipreading".  LipNet
maps a video of mouth-region frames to character probabilities using three
spatiotemporal convolutional blocks, a recurrent sequence model, and a final
linear/softmax layer trained with CTC.

This compact random-init reconstruction keeps the three 3D convolutional
feature extractors, temporal preservation, two bidirectional GRU layers, and
per-frame character classifier.  The spatial size and channel counts are small
enough for TorchLens graph rendering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STCNNBlock(nn.Module):
    """Spatiotemporal convolutional LipNet block."""

    def __init__(self, in_channels: int, out_channels: int, stride_hw: int = 2) -> None:
        """Initialize the 3D convolution block.

        Parameters
        ----------
        in_channels:
            Number of input video channels.
        out_channels:
            Number of output feature channels.
        stride_hw:
            Spatial max-pooling stride.
        """

        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 5, 5),
            padding=(1, 2, 2),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.stride_hw = stride_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, activation, and spatial pooling.

        Parameters
        ----------
        x:
            Video tensor of shape ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Feature video with preserved temporal length.
        """

        x = F.relu(self.bn(self.conv(x)))
        return F.max_pool3d(x, kernel_size=(1, self.stride_hw, self.stride_hw))


class LipNet(nn.Module):
    """Compact LipNet reconstruction."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 32) -> None:
        """Initialize LipNet.

        Parameters
        ----------
        vocab_size:
            Number of output CTC character classes, including blank.
        hidden_size:
            Hidden size of each bidirectional GRU direction.
        """

        super().__init__()
        self.stcnn = nn.Sequential(
            STCNNBlock(1, 8),
            STCNNBlock(8, 16),
            STCNNBlock(16, 24),
        )
        self.pre_rnn = nn.Linear(24 * 4 * 4, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Compute per-frame character log-probabilities.

        Parameters
        ----------
        video:
            Mouth-region video tensor of shape ``(batch, time, 1, height, width)``.

        Returns
        -------
        torch.Tensor
            Log probabilities with shape ``(time, batch, vocab_size)`` for CTC.
        """

        x = video.permute(0, 2, 1, 3, 4)
        x = self.stcnn(x)
        x = x.permute(0, 2, 1, 3, 4).flatten(2)
        x = torch.tanh(self.pre_rnn(x))
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return F.log_softmax(self.classifier(x), dim=-1).transpose(0, 1)


def build() -> nn.Module:
    """Build the compact LipNet model.

    Returns
    -------
    nn.Module
        Random-init LipNet in evaluation mode.
    """

    return LipNet().eval()


def example_input() -> torch.Tensor:
    """Return a small mouth video for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 8, 1, 32, 32)``.
    """

    return torch.randn(1, 8, 1, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "LipNet",
        "build",
        "example_input",
        "2016",
        "DC",
    ),
]
