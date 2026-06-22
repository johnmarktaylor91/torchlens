"""MetNet: neural weather nowcasting with ConvLSTM and axial attention.

Paper: "MetNet: A Neural Weather Model for Precipitation Forecasting",
Sønderby et al., 2020.

The compact reconstruction keeps the MetNet recipe: spatial downsampling of
multi-frame weather inputs, ConvLSTM temporal aggregation, axial self-attention
over the spatial grid, and precipitation-bin logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell."""

    def __init__(self, channels: int) -> None:
        """Initialize ConvLSTM gates."""

        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 4, 3, padding=1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one ConvLSTM step."""

        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class AxialAttention2d(nn.Module):
    """Row and column self-attention over a feature grid."""

    def __init__(self, channels: int) -> None:
        """Initialize axial attention modules."""

        super().__init__()
        self.row = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.col = nn.MultiheadAttention(channels, 4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply row then column attention."""

        batch, channels, height, width = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        rows, _ = self.row(rows, rows, rows)
        x = rows.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        cols = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        cols, _ = self.col(cols, cols, cols)
        return cols.reshape(batch, width, height, channels).permute(0, 3, 2, 1)


class MetNetCompact(nn.Module):
    """Compact MetNet nowcaster."""

    def __init__(self, channels: int = 24, bins: int = 8) -> None:
        """Initialize encoder, ConvLSTM, axial attention, and head."""

        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, channels, 3, stride=2, padding=1), nn.GELU())
        self.cell = ConvLSTMCell(channels)
        self.axial = AxialAttention2d(channels)
        self.head = nn.Conv2d(channels, bins, 1)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        """Predict categorical precipitation logits."""

        batch, steps, _, height, width = weather.shape
        h = torch.zeros(
            batch, 24, height // 2, width // 2, device=weather.device, dtype=weather.dtype
        )
        c = torch.zeros_like(h)
        for t in range(steps):
            h, c = self.cell(self.encoder(weather[:, t]), h, c)
        return self.head(self.axial(h))


def build() -> nn.Module:
    """Build compact MetNet."""

    return MetNetCompact()


def example_input() -> torch.Tensor:
    """Return a short multi-channel weather sequence."""

    return torch.randn(1, 4, 4, 32, 32)


MENAGERIE_ENTRIES = [("MetNet", "build", "example_input", "2020", "E7")]
