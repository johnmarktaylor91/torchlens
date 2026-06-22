"""NRTR-Transformer: no-recurrence Transformer scene-text recognizer.

Paper: Sheng et al. 2019, "NRTR: A No-Recurrence Sequence-to-Sequence Model
for Scene Text Recognition."  The compact model preserves the image-token
Transformer encoder and autoregressive Transformer decoder without an RNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import NRTRRecognizer, text_image


def build() -> nn.Module:
    """Build compact NRTR Transformer.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return NRTRRecognizer(with_modality_transform=False)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("NRTR-Transformer", "build", "example_input", "2019", "OCR")]
