"""KBNet-S denoising wrapper around the compact KBNet KBA implementation.

Zhang et al., 2023.
Paper: https://arxiv.org/abs/2303.02881
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.kbnet_l_deblur import build_kbnet_s_denoise as _build
from menagerie.classics.kbnet_l_deblur import example_input


def build_kbnet_s_denoise() -> nn.Module:
    """Build compact KBNet-S with kernel-basis attention for denoising."""

    return _build()


MENAGERIE_ENTRIES = [
    (
        "KBNet-S Denoise (kernel basis attention restoration)",
        "build_kbnet_s_denoise",
        "example_input",
        "2023",
        "image-restoration/denoising",
    ),
]
