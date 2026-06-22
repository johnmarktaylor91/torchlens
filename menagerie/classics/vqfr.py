"""VQFR: vector-quantized dictionary and parallel decoder face restoration.

Paper: "VQFR: Blind Face Restoration with Vector-Quantized Dictionary and
Parallel Decoder", Gu et al., ECCV 2022.

VQFR uses a VQ facial-detail dictionary and a parallel decoder: a texture branch
decodes dictionary features while a main branch preserves input fidelity and
interacts with texture through warping-like modulation.  This compact model
keeps the quantized dictionary lookup and dual decoder interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Soft vector-quantized facial dictionary lookup."""

    def __init__(self, codes: int = 64, channels: int = 32) -> None:
        """Initialize the VQ dictionary.

        Parameters
        ----------
        codes:
            Number of dictionary atoms.
        channels:
            Code dimension.
        """

        super().__init__()
        self.codebook = nn.Embedding(codes, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize features by nearest dictionary atoms.

        Parameters
        ----------
        x:
            Encoder feature map.

        Returns
        -------
        torch.Tensor
            Quantized dictionary feature map.
        """

        b, c, h, w = x.shape
        flat = x.flatten(2).transpose(1, 2)
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(flat, self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(dim=-1)
        )
        weights = torch.softmax(-distances, dim=-1)
        quantized = torch.matmul(weights, self.codebook.weight)
        return quantized.transpose(1, 2).view(b, c, h, w)


class TextureWarp(nn.Module):
    """Warp-like interaction from texture decoder to main decoder."""

    def __init__(self, channels: int) -> None:
        """Initialize texture interaction.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.offset_gate = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.mix = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, main: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
        """Fuse main and texture branches.

        Parameters
        ----------
        main:
            Main decoder feature.
        texture:
            Texture decoder feature.

        Returns
        -------
        torch.Tensor
            Fused main decoder feature.
        """

        gate = torch.sigmoid(self.offset_gate(torch.cat([main, texture], dim=1)))
        warped_texture = texture * gate
        return self.mix(torch.cat([main, warped_texture], dim=1))


class VQFRCompact(nn.Module):
    """Compact VQFR face-restoration model."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact VQFR.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.quantizer = VectorQuantizer(channels=channels)
        self.texture = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.main = nn.Conv2d(channels, channels, 3, padding=1)
        self.warp = TextureWarp(channels)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore a degraded aligned face crop.

        Parameters
        ----------
        x:
            Degraded RGB face crop.

        Returns
        -------
        torch.Tensor
            Restored RGB face crop.
        """

        feat = self.encoder(x)
        dictionary = self.quantizer(feat)
        texture = self.texture(dictionary)
        main = self.main(feat)
        fused = self.warp(main, texture)
        restored = self.decoder(fused)
        skip = F.interpolate(x, size=restored.shape[-2:], mode="bilinear", align_corners=False)
        return torch.tanh(restored + skip)


def build_vqfr() -> nn.Module:
    """Build compact VQFR.

    Returns
    -------
    nn.Module
        Random-init VQFR reconstruction.
    """

    return VQFRCompact()


def example_input() -> torch.Tensor:
    """Return a small degraded face crop.

    Returns
    -------
    torch.Tensor
        Example face tensor.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "VQFR (VQ dictionary parallel-decoder face restoration)",
        "build_vqfr",
        "example_input",
        "2022",
        "E7",
    )
]
