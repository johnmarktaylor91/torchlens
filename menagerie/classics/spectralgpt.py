"""SpectralGPT: spectral remote-sensing foundation model.

Paper: Hong et al. 2023/2024, "SpectralGPT: Spectral Remote Sensing Foundation
Model", arXiv:2311.07113 / TPAMI.

The compact model keeps SpectralGPT's key mechanism: 3D token generation over
spectral-spatial cubes followed by GPT-style causal transformer decoding and a
multi-target spectral reconstruction head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralGPT(nn.Module):
    """Compact 3D-token GPT for spectral image cubes."""

    def __init__(self, bands: int = 6, dim: int = 32, patches: int = 4) -> None:
        """Initialize 3D patch embedding and causal transformer blocks.

        Parameters
        ----------
        bands:
            Number of spectral input channels.
        dim:
            Token width.
        patches:
            Number of spatial tokens for the fixed example resolution.
        """

        super().__init__()
        self.patch = nn.Conv3d(1, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.pos = nn.Parameter(torch.zeros(1, (bands // 2) * patches, dim))
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.reconstruct = nn.Linear(dim, 2 * 4 * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct spectral-spatial patch targets from cube tokens.

        Parameters
        ----------
        x:
            Spectral image tensor with shape ``(batch, bands, height, width)``.

        Returns
        -------
        torch.Tensor
            Patch reconstruction tensor.
        """

        tokens = self.patch(x.unsqueeze(1)).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        mask = torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"), device=x.device).triu(
            1
        )
        hidden = self.blocks(tokens, mask=mask)
        return self.reconstruct(hidden)


def build() -> nn.Module:
    """Build a compact SpectralGPT model.

    Returns
    -------
    nn.Module
        Random-initialized SpectralGPT module.
    """

    return SpectralGPT()


def example_input() -> torch.Tensor:
    """Create a tiny spectral remote-sensing cube.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 6, 8, 8)``.
    """

    return torch.randn(1, 6, 8, 8)


MENAGERIE_ENTRIES = [
    ("SpectralGPT", "build", "example_input", "2023", "E6"),
]
