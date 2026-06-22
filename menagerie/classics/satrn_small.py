"""SATRN-Small: compact-width Self-Attention Text Recognition Network.

Paper: Lee et al. 2020, "On Recognizing Texts of Arbitrary Shapes with 2D
Self-Attention."  This variant keeps SATRN's 2D spatial self-attention encoder
and Transformer decoder with a smaller hidden width.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import SATRNRecognizer, text_image


def build() -> nn.Module:
    """Build compact SATRN-Small.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return SATRNRecognizer(dim=32)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("SATRN-Small", "build", "example_input", "2020", "OCR")]
