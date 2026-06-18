"""Compact Bilinear Pooling, 2016, Yang Gao et al.

Paper: Compact Bilinear Pooling.
Tensor Sketch approximates bilinear outer products with fixed Count-Sketch
hashes, FFT-domain convolution, signed-square-root, and L2 normalization.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CompactBilinearPooling(nn.Module):
    """Compact bilinear image classifier with Tensor Sketch pooling."""

    def __init__(
        self,
        channels: int = 8,
        sketch_dim: int = 64,
        num_classes: int = 5,
    ) -> None:
        """Initialize feature streams, sketches, and classifier.

        Parameters
        ----------
        channels:
            Feature channels in each stream.
        sketch_dim:
            Output dimension of the Count-Sketch projection.
        num_classes:
            Number of output classes.
        """
        super().__init__()
        self.feature_a = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
        self.feature_b = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
        self.sketch_dim = sketch_dim
        self.register_buffer("hash_a", torch.arange(channels) % sketch_dim)
        self.register_buffer("hash_b", (torch.arange(channels) * 3 + 1) % sketch_dim)
        sign_a = torch.where(torch.arange(channels) % 2 == 0, 1.0, -1.0)
        sign_b = torch.where(torch.arange(channels) % 3 == 0, -1.0, 1.0)
        self.register_buffer("sign_a", sign_a)
        self.register_buffer("sign_b", sign_b)
        self.classifier = nn.Linear(sketch_dim, num_classes)

    def _sketch(self, x: Tensor, hashes: Tensor, signs: Tensor) -> Tensor:
        """Project feature maps into Count-Sketch bins.

        Parameters
        ----------
        x:
            Feature map tensor with shape ``(B, C, H, W)``.
        hashes:
            Integer output bin per channel.
        signs:
            Fixed sign per channel.

        Returns
        -------
        Tensor
            Count-Sketch map with shape ``(B, D, H, W)``.
        """
        batch, channels, height, width = x.shape
        signed = x * signs.view(1, channels, 1, 1)
        out = x.new_zeros(batch, self.sketch_dim, height, width)
        index = hashes.view(1, channels, 1, 1).expand(batch, channels, height, width)
        return out.scatter_add(1, index, signed)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image using compact bilinear features.

        Parameters
        ----------
        x:
            RGB image tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        feat_a = self.feature_a(x)
        feat_b = self.feature_b(x)
        sketch_a = self._sketch(feat_a, self.hash_a, self.sign_a)
        sketch_b = self._sketch(feat_b, self.hash_b, self.sign_b)
        fft_a = torch.fft.rfft(sketch_a, dim=1)
        fft_b = torch.fft.rfft(sketch_b, dim=1)
        compact = torch.fft.irfft(fft_a * fft_b, n=self.sketch_dim, dim=1)
        pooled = compact.flatten(2).sum(dim=2)
        signed = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1.0e-8)
        return self.classifier(F.normalize(signed, dim=1))


def build() -> nn.Module:
    """Build a compact Tensor-Sketch bilinear classifier.

    Returns
    -------
    nn.Module
        Random-initialized compact bilinear module.
    """
    return CompactBilinearPooling()


def example_input() -> Tensor:
    """Return a traceable RGB image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)
