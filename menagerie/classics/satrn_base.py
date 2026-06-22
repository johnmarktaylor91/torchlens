"""SATRN-Base: Self-Attention Text Recognition Network.

Paper: Lee et al. 2020, "On Recognizing Texts of Arbitrary Shapes with 2D
Self-Attention."  The compact model keeps image patches with two-dimensional
position encoding, full spatial self-attention, and Transformer decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import SATRNRecognizer, text_image


def build() -> nn.Module:
    """Build compact SATRN-Base.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return SATRNRecognizer(dim=48)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("SATRN-Base", "build", "example_input", "2020", "OCR")]
