"""NRTR-ModalityTransform: Transformer OCR with 2D-to-1D image conversion.

Paper: Sheng et al. 2019, "NRTR: A No-Recurrence Sequence-to-Sequence Model
for Scene Text Recognition."  This compact model keeps the modality-transform
block that turns 2D image features into a 1D token sequence before the
Transformer encoder-decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import NRTRRecognizer, text_image


def build() -> nn.Module:
    """Build compact NRTR with modality transform.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return NRTRRecognizer(with_modality_transform=True)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("NRTR-ModalityTransform", "build", "example_input", "2019", "OCR")]
