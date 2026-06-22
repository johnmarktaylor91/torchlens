"""SatMAE: masked autoencoder for temporal and multispectral satellite data.

Paper: Cong et al. 2022, "SatMAE: Pre-training Transformers for Temporal and
Multi-Spectral Satellite Imagery" (NeurIPS).

The compact reconstruction uses tubelet patching, independent temporal/spectral
position embeddings, an asymmetric MAE encoder/decoder, and reconstruction of
masked satellite patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SatMAE(nn.Module):
    """Small SatMAE-style masked autoencoder."""

    def __init__(self, channels: int = 4, times: int = 3, dim: int = 32) -> None:
        """Initialize patch embedding and asymmetric transformer stacks.

        Parameters
        ----------
        channels:
            Number of multispectral channels.
        times:
            Number of temporal observations.
        dim:
            Token width.
        """

        super().__init__()
        self.patch = nn.Conv2d(channels, dim, kernel_size=4, stride=4)
        self.time_pos = nn.Parameter(torch.zeros(1, times, 1, dim))
        self.space_pos = nn.Parameter(torch.zeros(1, 1, 4, dim))
        enc = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        dec = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.decoder = nn.TransformerEncoder(dec, num_layers=1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.head = nn.Linear(dim, channels * 4 * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visible temporal patches and decode all patch targets.

        Parameters
        ----------
        x:
            Satellite tensor with shape ``(batch, time, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Reconstructed patch pixels for all time/space tokens.
        """

        batch, steps, channels, height, width = x.shape
        frames = x.reshape(batch * steps, channels, height, width)
        tokens = self.patch(frames).flatten(2).transpose(1, 2).reshape(batch, steps, 4, -1)
        tokens = tokens + self.time_pos[:, :steps] + self.space_pos
        seq = tokens.reshape(batch, steps * 4, -1)
        visible = seq[:, ::2]
        encoded = self.encoder(visible)
        full = self.mask_token.expand(batch, seq.shape[1], -1).clone()
        full[:, ::2] = encoded
        decoded = self.decoder(full)
        return self.head(decoded)


def build() -> nn.Module:
    """Build a compact SatMAE model.

    Returns
    -------
    nn.Module
        Random-initialized SatMAE module.
    """

    return SatMAE()


def example_input() -> torch.Tensor:
    """Create a small temporal multispectral image.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 3, 4, 8, 8)``.
    """

    return torch.randn(1, 3, 4, 8, 8)


MENAGERIE_ENTRIES = [
    ("SatMAE", "build", "example_input", "2022", "E5"),
]
