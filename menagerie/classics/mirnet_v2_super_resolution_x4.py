"""MIRNet-v2 x4 super-resolution wrapper.

Zamir et al., 2022.
Paper: https://arxiv.org/abs/2205.01649
"""

from __future__ import annotations

import torch.nn as nn

from menagerie.classics.mirnet_original import build_mirnet_v2_super_resolution_x4 as _build
from menagerie.classics.mirnet_original import example_input


def build_mirnet_v2_super_resolution_x4() -> nn.Module:
    """Build compact MIRNet-v2 x4 model with pixel-shuffle reconstruction."""

    return _build()


MENAGERIE_ENTRIES = [
    (
        "MIRNet-v2 Super-Resolution x4 (recursive MRB plus pixel shuffle)",
        "build_mirnet_v2_super_resolution_x4",
        "example_input",
        "2022",
        "image-restoration/super-resolution",
    ),
]
