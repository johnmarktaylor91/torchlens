"""MASTER-ResNet31: Multi-Aspect Non-local scene-text recognizer.

Paper: Lu et al. 2019/2021, "MASTER: Multi-Aspect Non-local Network for Scene
Text Recognition."  This compact reconstruction keeps the ResNet31 visual
encoder, multi-aspect global-context attention, and Transformer-style
self-attention encoder-decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import MASTERRecognizer, text_image


def build() -> nn.Module:
    """Build compact MASTER-ResNet31.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return MASTERRecognizer(extra=False)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("MASTER-ResNet31", "build", "example_input", "2019", "OCR")]
