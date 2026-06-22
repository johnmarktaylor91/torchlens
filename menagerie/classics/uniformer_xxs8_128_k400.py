"""UniFormer-XXS/8 for Kinetics-400 video classification.

Paper: UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning
and UniFormer: Unifying Convolution and Self-attention for Visual Recognition, 2022.

This module reuses the compact UniFormer video implementation and changes the clip
length/depth profile to represent the eight-frame XXS Kinetics-400 variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.uniformer_xxs4_160_k400 import CompactVideoUniFormer


def build_uniformer_xxs8_128_k400() -> nn.Module:
    """Build a compact UniFormer-XXS/8 Kinetics-400 classifier.

    Returns
    -------
    nn.Module
        Random-init compact UniFormer model.
    """

    return CompactVideoUniFormer(depths=(1, 1, 2, 2))


def example_input() -> torch.Tensor:
    """Create a compact eight-frame video input.

    Returns
    -------
    torch.Tensor
        Input tensor with shape ``(1, 3, 8, 32, 32)``.
    """

    return torch.randn(1, 3, 8, 32, 32)


build = build_uniformer_xxs8_128_k400

MENAGERIE_ENTRIES = [
    (
        "uniformer_xxs8_128_k400",
        "build_uniformer_xxs8_128_k400",
        "example_input",
        "2022",
        "E5",
    ),
]
