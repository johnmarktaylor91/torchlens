"""Restormer dual-pixel defocus deblurring.

Paper: "Restormer: Efficient Transformer for High-Resolution Image Restoration",
Zamir et al., CVPR 2022.

For dual-pixel defocus, Restormer consumes paired left/right sub-aperture views
as a six-channel input while keeping its defining MDTA channel-attention and
GDFN gated depthwise transformer blocks.  The compact reconstruction preserves
that six-channel-to-RGB restoration contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.restormer_deraining import RestormerCompact


def build() -> nn.Module:
    """Build compact dual-pixel Restormer."""

    return RestormerCompact(channels=24, in_channels=6)


def example_input() -> torch.Tensor:
    """Return paired dual-pixel RGB views as a six-channel image."""

    return torch.randn(1, 6, 24, 24)


MENAGERIE_ENTRIES = [
    ("Restormer dual-pixel defocus (6-channel MDTA/GDFN)", "build", "example_input", "2022", "E5")
]
