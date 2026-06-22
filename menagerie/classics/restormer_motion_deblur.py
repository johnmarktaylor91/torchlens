"""Restormer motion deblurring: MDTA/GDFN restoration transformer.

Paper: "Restormer: Efficient Transformer for High-Resolution Image Restoration",
Zamir et al., CVPR 2022.

The motion-deblurring configuration uses the same Restormer building blocks as
the paper's deraining/denoising variants: channel-transposed MDTA attention,
GDFN gated depthwise feed-forward layers, and encoder-decoder restoration with
a residual image prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.restormer_deraining import RestormerCompact


def build() -> nn.Module:
    """Build compact Restormer motion deblurring model."""

    return RestormerCompact(channels=24, in_channels=3)


def example_input() -> torch.Tensor:
    """Return a small motion-blurred RGB image tensor."""

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [
    ("Restormer motion deblur (MDTA + GDFN transformer)", "build", "example_input", "2022", "E5")
]
