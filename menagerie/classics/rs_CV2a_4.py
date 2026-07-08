# SOURCE: vendored from bootphon/articulatory_inversion @ HEAD (Training/model.py)
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class MyAc2ArtModel(nn.Module):
    """Acoustic-to-articulatory BLSTM model from the official PyTorch source."""

    def __init__(
        self,
        input_dim: int = 13,
        output_dim: int = 6,
        hidden_dim: int = 8,
        batch_norm: bool = True,
    ) -> None:
        """Initialize the dense, BLSTM, normalization, and readout layers."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
        self.lstm_layer = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.batch_norm_layer = nn.BatchNorm1d(hidden_dim * 2)
        self.lstm_layer_2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.batch_norm_layer_2 = nn.BatchNorm1d(hidden_dim * 2)
        self.readout_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: Tensor, filter_output: bool | None = None) -> Tensor:
        """Run the official dense-BLSTM-BLSTM-readout forward path."""
        del filter_output
        dense_out = F.relu(self.first_layer(x))
        dense_out_2 = F.relu(self.second_layer(dense_out))
        lstm_out, _ = self.lstm_layer(dense_out_2)
        if self.batch_norm:
            batch_size = x.shape[0]
            lstm_out_temp = lstm_out.contiguous().view(batch_size, 2 * self.hidden_dim, -1)
            lstm_out_temp = F.relu(self.batch_norm_layer(lstm_out_temp))
            lstm_out = lstm_out_temp.view(batch_size, -1, 2 * self.hidden_dim)
        lstm_out = F.relu(lstm_out)
        lstm_out, _ = self.lstm_layer_2(lstm_out)
        if self.batch_norm:
            batch_size = x.shape[0]
            lstm_out_temp = lstm_out.contiguous().view(batch_size, 2 * self.hidden_dim, -1)
            lstm_out_temp = F.relu(self.batch_norm_layer_2(lstm_out_temp))
            lstm_out = lstm_out_temp.view(batch_size, -1, 2 * self.hidden_dim)
        lstm_out = F.relu(lstm_out)
        return self.readout_layer(lstm_out)


def build_articulatory_blstm() -> MyAc2ArtModel:
    """Build a traceable articulatory-acoustic BLSTM."""
    return MyAc2ArtModel()


def example_input_articulatory_blstm() -> Tensor:
    """Return a short acoustic feature sequence."""
    return torch.randn(2, 12, 13)


MENAGERIE_ENTRIES = [
    (
        "Articulatory-Acoustic BLSTM (Acoustic-to-Articulatory Mapping)",
        build_articulatory_blstm,
        example_input_articulatory_blstm,
        2017,
        "CV2a-articulatory-blstm",
    ),
]
