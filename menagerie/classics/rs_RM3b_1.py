# FAITHFUL REIMPLEMENTATION from detailed description (no public code): a hybrid
# 1D-CNN + LSTM acoustic-emission (AE) source-localization network as described in
# W. Fu, R. Zhou, Y. Gao, Z. Guo, Q. Yu, "Damage Source Localization in Concrete Slabs
# Based on Acoustic Emission and Machine Learning", IEEE Sensors J. (2025) -- lightweight
# DNN/1D-CNN/LSTM models for planar AE source localization from a sensor array; and
# J. Zuo, B. Sheil, S. Acikgoz, "A three-stage approach based on 1D-CNNs for AE source
# localisation on historic fibrous plaster ceilings" (2024). No unified public repository
# exists for this architecture family; this module reimplements the commonly-described
# mechanism: per-sensor 1D-CNN waveform feature extraction, followed by an LSTM that
# fuses features across the sensor array, followed by a small regression head that
# outputs the 2D planar source coordinates.
from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class AECNNLSTMLocalizer(nn.Module):
    """CNN-LSTM acoustic-emission source localizer.

    A shared 1D-CNN extracts local waveform features independently for each sensor
    channel in an AE sensor array; the per-sensor feature vectors are then treated as
    a sequence (one step per sensor) and fused by an LSTM, whose final hidden state
    feeds a small MLP regression head that predicts the 2D planar source location.
    """

    def __init__(
        self,
        cnn_channels: tuple[int, ...] = (8, 16, 32),
        lstm_hidden: int = 32,
        out_dim: int = 2,
    ) -> None:
        """Initialize the per-sensor CNN feature extractor, LSTM fusion, and MLP head."""
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c_out in cnn_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ]
            c_in = c_out
        self.feature_extractor = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=cnn_channels[-1], hidden_size=lstm_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Localize an AE source from a multi-sensor raw waveform array.

        Parameters
        ----------
        x
            Raw AE waveform tensor of shape ``(batch, num_sensors, length)``.

        Returns
        -------
        Tensor
            Predicted planar source coordinates of shape ``(batch, out_dim)``.
        """
        n, s, length = x.shape
        x = x.reshape(n * s, 1, length)
        feats = self.feature_extractor(x)
        feats = feats.mean(dim=-1)
        feats = feats.reshape(n, s, -1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        return self.head(out)


def build_ae_cnn_lstm_localizer() -> AECNNLSTMLocalizer:
    """Build a traceable AE CNN-LSTM source localizer at tiny size."""
    return AECNNLSTMLocalizer(cnn_channels=(8, 16, 32), lstm_hidden=32, out_dim=2)


def example_input_ae_cnn_lstm_localizer() -> Tensor:
    """Return a 4-sensor raw AE waveform example."""
    return torch.randn(2, 4, 512)


MENAGERIE_ENTRIES = [
    (
        "Acoustic Emission CNN-LSTM (AE source localization)",
        build_ae_cnn_lstm_localizer,
        example_input_ae_cnn_lstm_localizer,
        2025,
        "RM3b-ae-cnn-lstm",
    ),
]
