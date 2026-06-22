"""RobustScanner-ResNet31-Attn: RobustScanner with attention-refined features.

Paper: Yue et al. 2020, "RobustScanner: Dynamically Enhancing Positional
Clues for Robust Text Recognition."  This variant keeps the same hybrid and
position branches, with an added global-context attention refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import RobustScannerRecognizer, text_image


def build() -> nn.Module:
    """Build compact RobustScanner-ResNet31-Attn.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return RobustScannerRecognizer(attention_backbone=True)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("RobustScanner-ResNet31-Attn", "build", "example_input", "2020", "OCR")]
