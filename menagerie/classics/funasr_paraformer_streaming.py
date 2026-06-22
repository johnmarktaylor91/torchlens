"""FunASR streaming Paraformer compact reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.funasr_paraformer_large import CompactParaformer


def build_funasr_paraformer_streaming() -> nn.Module:
    """Build compact streaming Paraformer with causal SAN-M encoder blocks.

    Returns
    -------
    nn.Module
        Random-init compact streaming Paraformer.
    """

    return CompactParaformer(causal=True)


def example_input() -> torch.Tensor:
    """Create compact streaming speech features.

    Returns
    -------
    torch.Tensor
        Feature tensor of shape ``(1, 24, 32)``.
    """

    return torch.randn(1, 24, 32)


build = build_funasr_paraformer_streaming

MENAGERIE_ENTRIES = [
    (
        "funasr_paraformer_streaming",
        "build_funasr_paraformer_streaming",
        "example_input",
        "2022",
        "E5",
    ),
]
