"""NAFNet width64 GoPro deblurring: nonlinear-activation-free restoration.

Paper: "Simple Baselines for Image Restoration", Chen et al., ECCV 2022.

The GoPro deblurring model is the wider NAFNet setting: LayerNorm, depthwise
convolution, SimpleGate multiplicative channel splits, Simplified Channel
Attention, and learned residual scaling without ReLU/GELU/Sigmoid activations in
the main block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.nafnet_width32_sidd_denoise import NAFNetCompact


def build() -> nn.Module:
    """Build compact width64 NAFNet GoPro deblurring model."""

    return NAFNetCompact(width=64)


def example_input() -> torch.Tensor:
    """Return a small motion-blurred RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    ("NAFNet width64 GoPro deblur (SimpleGate + SCA)", "build", "example_input", "2022", "E5")
]
