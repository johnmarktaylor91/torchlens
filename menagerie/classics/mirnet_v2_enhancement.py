"""MIRNet-v2 enhancement wrapper using compact multi-scale residual blocks.

Zamir et al., 2022.
Paper: https://arxiv.org/abs/2205.01649
"""

from __future__ import annotations

import torch.nn as nn

from menagerie.classics.mirnet_original import build_mirnet_v2_enhancement as _build
from menagerie.classics.mirnet_original import example_input


def build_mirnet_v2_enhancement() -> nn.Module:
    """Build compact MIRNet-v2 enhancement model."""

    return _build()


MENAGERIE_ENTRIES = [
    (
        "MIRNet-v2 enhancement (recursive multi-scale residual restoration)",
        "build_mirnet_v2_enhancement",
        "example_input",
        "2022",
        "image-restoration/enhancement",
    ),
]
