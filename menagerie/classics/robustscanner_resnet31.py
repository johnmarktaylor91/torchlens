"""RobustScanner-ResNet31: dynamic position/context OCR decoder.

Paper: Yue et al. 2020, "RobustScanner: Dynamically Enhancing Positional
Clues for Robust Text Recognition."  The compact model keeps the ResNet31
encoder, hybrid attention decoder, position-enhancement branch, and dynamic
fusion of context and positional logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import RobustScannerRecognizer, text_image


def build() -> nn.Module:
    """Build compact RobustScanner-ResNet31.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return RobustScannerRecognizer(attention_backbone=False)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("RobustScanner-ResNet31", "build", "example_input", "2020", "OCR")]
