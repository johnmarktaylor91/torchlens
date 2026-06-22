"""SSFTT hyperspectral transformer compact random-init reconstruction.

Paper: Spectral-Spatial Feature Tokenization Transformer for Hyperspectral
Image Classification (Sun, Zhao, Zheng, Wu, IEEE TGRS 2022).

The distinctive pipeline is shallow spectral-spatial extraction with 3D then 2D
convolutions, Gaussian-weighted feature tokenization, transformer encoding, and
classification from learned semantic tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GaussianTokenizer(nn.Module):
    """Gaussian weighted feature tokenizer used by SSFTT."""

    def __init__(self, dim: int = 32, tokens: int = 6) -> None:
        """Initialize learnable token centers and widths."""

        super().__init__()
        self.centers = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(tokens))

    def forward(self, features: Tensor) -> Tensor:
        """Convert spatial-spectral pixels into weighted semantic tokens."""

        pixels = features.flatten(2).transpose(1, 2)
        diff = pixels[:, :, None, :] - self.centers[None, None, :, :]
        sigma = torch.exp(self.log_sigma)[None, None, :, None].clamp_min(0.05)
        weights = torch.softmax(-(diff.pow(2) / sigma).sum(dim=-1), dim=1)
        return torch.einsum("bnt,bnd->btd", weights, pixels)


class SSFTT(nn.Module):
    """Compact spectral-spatial feature tokenization transformer."""

    def __init__(self, classes: int = 9, dim: int = 32) -> None:
        """Initialize 3D/2D convolutions, tokenizer, transformer, and classifier."""

        super().__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, (5, 3, 3), padding=(2, 1, 1)), nn.ReLU())
        self.conv2d = nn.Sequential(nn.Conv2d(8 * 12, dim, 3, padding=1), nn.ReLU())
        self.tokenizer = GaussianTokenizer(dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.cls = nn.Linear(dim, classes)

    def forward(self, cube: Tensor) -> Tensor:
        """Classify a small hyperspectral cube."""

        x = self.conv3d(cube.unsqueeze(1))
        bsz, channels, bands, height, width = x.shape
        x = x.reshape(bsz, channels * bands, height, width)
        x = self.conv2d(x)
        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        return self.cls(encoded.mean(dim=1))


def build() -> nn.Module:
    """Build a compact random-init SSFTT model."""

    return SSFTT().eval()


def example_input() -> Tensor:
    """Return a small hyperspectral patch with 12 spectral bands."""

    return torch.randn(1, 12, 9, 9)


MENAGERIE_ENTRIES = [
    ("ssftt_hsi", "build", "example_input", "2022", "DC"),
]
