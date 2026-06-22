"""SAR-ResNet31: Show-Attend-and-Read irregular text recognizer.

Paper: Li et al. 2019, "A Simple and Strong Baseline for Irregular Text
Recognition."  This compact classic keeps the ResNet31 visual encoder,
holistic LSTM encoding, LSTM decoder, and 2D attention over visual features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import SARRecognizer, text_image


def build() -> nn.Module:
    """Build compact SAR-ResNet31.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return SARRecognizer()


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("SAR-ResNet31", "build", "example_input", "2019", "OCR")]
