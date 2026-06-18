"""MD-LSTM / Pyramidal-LSTM, 2007, Alex Graves and Jurgen Schmidhuber.

Paper: Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks.
Each 2D LSTM cell receives recurrent inputs from vertical and horizontal
predecessors with separate forget gates, then four corner sweeps are summed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MDLSTMSweep(nn.Module):
    """Single directional two-dimensional LSTM sweep."""

    def __init__(self, input_channels: int, hidden_size: int) -> None:
        """Initialize multidimensional LSTM gates.

        Parameters
        ----------
        input_channels:
            Number of input image channels.
        hidden_size:
            Hidden channels per cell.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_channels, 5 * hidden_size)
        self.up_proj = nn.Linear(hidden_size, 5 * hidden_size, bias=False)
        self.left_proj = nn.Linear(hidden_size, 5 * hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Scan an image from top-left to bottom-right.

        Parameters
        ----------
        x:
            Image tensor ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Hidden map ``(B, hidden_size, H, W)``.
        """
        batch, _, height, width = x.shape
        zero_h = x.new_zeros(batch, self.hidden_size)
        zero_c = x.new_zeros(batch, self.hidden_size)
        hidden_rows: list[list[Tensor]] = []
        cell_rows: list[list[Tensor]] = []
        for row in range(height):
            hidden_row: list[Tensor] = []
            cell_row: list[Tensor] = []
            for col in range(width):
                h_up = hidden_rows[row - 1][col] if row > 0 else zero_h
                c_up = cell_rows[row - 1][col] if row > 0 else zero_c
                h_left = hidden_row[col - 1] if col > 0 else zero_h
                c_left = cell_row[col - 1] if col > 0 else zero_c
                pixel = x[:, :, row, col]
                gates = self.input_proj(pixel) + self.up_proj(h_up) + self.left_proj(h_left)
                i_gate, o_gate, f_up, f_left, candidate = gates.chunk(5, dim=1)
                cell = (
                    torch.sigmoid(i_gate) * torch.tanh(candidate)
                    + torch.sigmoid(f_up) * c_up
                    + torch.sigmoid(f_left) * c_left
                )
                hidden = torch.sigmoid(o_gate) * torch.tanh(cell)
                hidden_row.append(hidden)
                cell_row.append(cell)
            hidden_rows.append(hidden_row)
            cell_rows.append(cell_row)
        rows = [torch.stack(hidden_row, dim=2) for hidden_row in hidden_rows]
        return torch.stack(rows, dim=2).permute(0, 1, 3, 2)


class MDLSTM2D(nn.Module):
    """Four-direction MD-LSTM image encoder."""

    def __init__(self, input_channels: int = 1, hidden_size: int = 4, output_size: int = 6) -> None:
        """Initialize four corner sweeps and classifier.

        Parameters
        ----------
        input_channels:
            Number of image channels.
        hidden_size:
            Hidden channels per directional sweep.
        output_size:
            Output feature size.
        """
        super().__init__()
        self.sweep = MDLSTMSweep(input_channels, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Encode image context with four MD-LSTM corner sweeps.

        Parameters
        ----------
        x:
            Image tensor ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Global output features.
        """
        y1 = self.sweep(x)
        y2 = torch.flip(self.sweep(torch.flip(x, dims=(3,))), dims=(3,))
        y3 = torch.flip(self.sweep(torch.flip(x, dims=(2,))), dims=(2,))
        y4 = torch.flip(self.sweep(torch.flip(x, dims=(2, 3))), dims=(2, 3))
        pooled = (y1 + y2 + y3 + y4).mean(dim=(2, 3))
        return self.output(pooled)


def build() -> nn.Module:
    """Build a compact four-direction MD-LSTM.

    Returns
    -------
    nn.Module
        Random-initialized MDLSTM2D.
    """
    return MDLSTM2D()


def example_input() -> Tensor:
    """Return a traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 5, 5)``.
    """
    return torch.randn(1, 1, 5, 5)
