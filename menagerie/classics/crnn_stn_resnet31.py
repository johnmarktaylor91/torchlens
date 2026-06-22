"""CRNN-STN-ResNet31: rectified CTC scene-text recognizer.

Paper: Baek et al. 2019, "What Is Wrong With Scene Text Recognition Model
Comparisons?"  The compact classic keeps the TPS/STN rectification stage,
ResNet31-style visual features, bidirectional LSTM sequence model, and CTC head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import CRNNSTNResNet31, text_image


def build() -> nn.Module:
    """Build compact CRNN-STN-ResNet31.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return CRNNSTNResNet31()


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("CRNN-STN-ResNet31", "build", "example_input", "2019", "OCR")]
