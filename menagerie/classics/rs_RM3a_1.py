# FAITHFUL REIMPLEMENTATION from Esfahani, Wang, Wu & Yuan, "AbolDeepIO: A Novel Deep
# Inertial Odometry Network for Autonomous Vehicles," IEEE Trans. Intelligent Transportation
# Systems, 21(5):1941-1950, 2020 (no public code found; gh code/repo search and web search both
# empty). Description source: paper abstract ("novel triple-channel deep IO network... considers
# the time interval between two consecutive IMU readings") plus Chen & Pan, "Deep Learning for
# Inertial Positioning: A Survey" (arXiv:2303.03757), which describes AbolDeepIO as "an improved
# triple-channel LSTM network that predicts polar vectors ... built on IONet [Chen et al. 2018]".
# IONet's polar-vector head (distance + heading-change over a sliding window of windowed 6-axis
# IMU readings) is the well-documented target formulation AbolDeepIO extends with a third,
# dedicated LSTM channel for the inter-sample time interval (dt) alongside the accelerometer and
# gyroscope channels, fusing all three before the polar-vector regression head.
#
# FAITHFUL REIMPLEMENTATION from Chen, Peng, Xie, Zhang, Li & Liu, "ACDIN: Bridging the gap
# between artificial and real bearing damages for bearing fault diagnosis," Neurocomputing,
# 294:61-71, 2018 (no public code found). Description source: paper title/abstract as quoted by
# multiple citing surveys (e.g. Jiao et al., "A comprehensive review on convolutional neural
# network in machine fault diagnosis," arXiv:2002.07605; Li et al., IEEE Access 2020) --
# "a deep convolutional structure ... based on 1D-CNNs, which improved the feature extraction
# ability by adding an inception layer, with 1D convolution improved by Atrous convolution,"
# operating directly on raw time-domain vibration signal (not a time-frequency image) to close
# the domain gap between artificially-seeded and real bearing damage.
#
# FAITHFUL REIMPLEMENTATION of a generic acoustic-emission (AE) hit-sequence LSTM classifier as
# described across AE-based structural-health-monitoring literature (no single canonical public
# repo located): AE events are summarized per-hit by standard AE parameters -- rise time, energy,
# counts, and duration -- and the resulting per-hit feature sequence is classified with a
# standard stacked LSTM encoder followed by a fully-connected classification head.
from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class AbolDeepIO(nn.Module):
    """Triple-channel LSTM inertial-odometry network (accelerometer / gyroscope / dt channels).

    Each IMU-derived channel is encoded by its own bidirectional LSTM branch; the branches'
    final hidden states are fused and regressed to a 2-D polar vector (distance, heading
    change) over the input window, following IONet's polar-vector odometry formulation that
    AbolDeepIO builds on with the dedicated third (time-interval) channel.
    """

    def __init__(
        self,
        acc_hidden: int = 24,
        gyro_hidden: int = 24,
        dt_hidden: int = 8,
        fusion_hidden: int = 32,
    ) -> None:
        """Initialize the three per-channel LSTM branches and the fusion regression head."""
        super().__init__()
        self.acc_branch = nn.LSTM(
            input_size=3, hidden_size=acc_hidden, num_layers=2, batch_first=True, bidirectional=True
        )
        self.gyro_branch = nn.LSTM(
            input_size=3,
            hidden_size=gyro_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.dt_branch = nn.LSTM(
            input_size=1, hidden_size=dt_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        fused_dim = 2 * (acc_hidden + gyro_hidden + dt_hidden)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 2),  # polar vector: (distance, heading change)
        )

    def forward(self, acc: Tensor, gyro: Tensor, dt: Tensor) -> Tensor:
        """Encode acc/gyro/dt windows and regress the fused polar-vector displacement.

        Parameters
        ----------
        acc
            Accelerometer window, shape (batch, seq_len, 3).
        gyro
            Gyroscope window, shape (batch, seq_len, 3).
        dt
            Inter-sample time-interval window, shape (batch, seq_len, 1).

        Returns
        -------
        Tensor
            Polar vector (distance, heading change), shape (batch, 2).
        """
        _, (acc_h, _) = self.acc_branch(acc)
        _, (gyro_h, _) = self.gyro_branch(gyro)
        _, (dt_h, _) = self.dt_branch(dt)

        # Concatenate the last-layer forward/backward hidden states from each branch.
        acc_feat = torch.cat([acc_h[-2], acc_h[-1]], dim=-1)
        gyro_feat = torch.cat([gyro_h[-2], gyro_h[-1]], dim=-1)
        dt_feat = torch.cat([dt_h[-2], dt_h[-1]], dim=-1)

        fused = torch.cat([acc_feat, gyro_feat, dt_feat], dim=-1)
        return self.fusion_fc(fused)


class AtrousInceptionBlock1d(nn.Module):
    """Inception-style block with parallel atrous (dilated) 1-D convolution branches."""

    def __init__(self, in_channels: int, branch_channels: int) -> None:
        """Initialize the four parallel dilated/pooling branches of the block."""
        super().__init__()
        self.branch_d1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Conv1d(branch_channels * 4, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the four dilated/pooling branches, concatenate, and project back to width."""
        cat = torch.cat(
            [self.branch_d1(x), self.branch_d2(x), self.branch_d4(x), self.branch_pool(x)], dim=1
        )
        return self.project(cat)


class ACDIN(nn.Module):
    """Deep inception net with atrous convolution for raw-signal bearing fault diagnosis."""

    def __init__(self, num_classes: int = 4, width: int = 16) -> None:
        """Initialize the wide first-layer stem, stacked atrous-inception blocks, and head."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            AtrousInceptionBlock1d(width, width),
            nn.MaxPool1d(kernel_size=2, stride=2),
            AtrousInceptionBlock1d(width, width),
            nn.MaxPool1d(kernel_size=2, stride=2),
            AtrousInceptionBlock1d(width, width),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the ACDIN stem, atrous-inception blocks, and classification head.

        Parameters
        ----------
        x
            Raw 1-D vibration signal window, shape (batch, 1, length).

        Returns
        -------
        Tensor
            Fault-class logits, shape (batch, num_classes).
        """
        feat = self.blocks(self.stem(x))
        pooled = self.gap(feat).squeeze(-1)
        return self.classifier(pooled)


class AcousticEmissionLSTM(nn.Module):
    """Generic stacked-LSTM classifier over acoustic-emission hit feature sequences."""

    def __init__(self, num_features: int = 4, hidden_size: int = 32, num_classes: int = 3) -> None:
        """Initialize the LSTM encoder and the final-hidden-state classification head."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features, hidden_size=hidden_size, num_layers=2, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hits: Tensor) -> Tensor:
        """Classify a sequence of AE hit feature vectors.

        Parameters
        ----------
        hits
            AE hit features (rise time, energy, counts, duration) per hit,
            shape (batch, seq_len, num_features).

        Returns
        -------
        Tensor
            Class logits, shape (batch, num_classes).
        """
        _, (h_n, _) = self.lstm(hits)
        return self.classifier(h_n[-1])


def build_aboldeepio() -> AbolDeepIO:
    """Build a traceable AbolDeepIO instance."""
    return AbolDeepIO()


def example_input_aboldeepio() -> tuple[Tensor, Tensor, Tensor]:
    """Return acc/gyro/dt window tensors for AbolDeepIO."""
    acc = torch.randn(1, 20, 3)
    gyro = torch.randn(1, 20, 3)
    dt = torch.rand(1, 20, 1) * 0.01
    return acc, gyro, dt


def build_acdin() -> ACDIN:
    """Build a traceable ACDIN instance."""
    return ACDIN()


def example_input_acdin() -> Tensor:
    """Return a raw vibration-signal window for ACDIN."""
    return torch.randn(1, 1, 1024)


def build_ae_lstm() -> AcousticEmissionLSTM:
    """Build a traceable acoustic-emission LSTM classifier instance."""
    return AcousticEmissionLSTM()


def example_input_ae_lstm() -> Tensor:
    """Return an AE hit-feature sequence (rise time, energy, counts, duration)."""
    return torch.randn(1, 12, 4)


MENAGERIE_ENTRIES = [
    ("AbolDeepIO", build_aboldeepio, example_input_aboldeepio, 2019, "RM3a-aboldeepio"),
    ("ACDIN", build_acdin, example_input_acdin, 2018, "RM3a-acdin"),
    ("Acoustic Emission AE-LSTM", build_ae_lstm, example_input_ae_lstm, 2020, "RM3a-ae-lstm"),
]
