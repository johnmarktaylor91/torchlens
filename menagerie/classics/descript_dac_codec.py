"""Descript DAC codec alias compact reconstruction.

Paper: High-Fidelity Audio Compression with Improved RVQGAN
(Kumar et al., 2023).

This dependency-gated catalog target refers to the same DAC codec family:
strided convolutional audio encoder, residual vector quantizer, Snake
activations, and convolutional decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dac_descript import DescriptDAC


def build() -> nn.Module:
    """Build a compact random-init Descript DAC codec."""

    return DescriptDAC(channels=24).eval()


def example_input() -> Tensor:
    """Return a short mono waveform."""

    return torch.randn(1, 1, 256)


MENAGERIE_ENTRIES = [
    ("descript_dac_codec", "build", "example_input", "2023", "DC"),
]
