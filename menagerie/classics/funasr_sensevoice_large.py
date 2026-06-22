"""FunASR SenseVoice large compact reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.funasr_sensevoice_small import CompactSenseVoice


def build_funasr_sensevoice_large() -> nn.Module:
    """Build compact SenseVoice large with a deeper encoder profile.

    Returns
    -------
    nn.Module
        Random-init compact SenseVoice.
    """

    return CompactSenseVoice(layers=3)


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact SenseVoice large inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Speech features, language ID, and task ID.
    """

    return torch.randn(1, 16, 40), torch.tensor([1]), torch.tensor([0])


build = build_funasr_sensevoice_large

MENAGERIE_ENTRIES = [
    ("funasr_sensevoice_large", "build_funasr_sensevoice_large", "example_input", "2024", "E6"),
]
