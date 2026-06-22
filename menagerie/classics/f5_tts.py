"""F5-TTS compact top-level reconstruction.

Paper: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching,
2024.  The public F5-TTS model is represented here by the compact DiT flow-matching
decoder in :mod:`menagerie.classics.f5_tts_dit`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.f5_tts_dit import build_f5_tts_dit
from menagerie.classics.f5_tts_dit import example_input as _dit_example_input


def build_f5_tts() -> nn.Module:
    """Build compact F5-TTS.

    Returns
    -------
    nn.Module
        Random-init compact F5-TTS model.
    """

    return build_f5_tts_dit()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact F5-TTS inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Noisy mel, text tokens, reference mel, mask, and time.
    """

    return _dit_example_input()


build = build_f5_tts

MENAGERIE_ENTRIES = [
    ("F5-TTS", "build_f5_tts", "example_input", "2024", "E6"),
]
