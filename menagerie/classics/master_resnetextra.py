"""MASTER-ResNetExtra: deeper MASTER text recognizer.

Paper: Lu et al. 2019/2021, "MASTER: Multi-Aspect Non-local Network for Scene
Text Recognition."  The model keeps MASTER's multi-aspect global-context
attention and Transformer decoder, with an extra residual refinement stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import MASTERRecognizer, text_image


def build() -> nn.Module:
    """Build compact MASTER-ResNetExtra.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return MASTERRecognizer(extra=True)


def example_input() -> torch.Tensor:
    """Create a compact word image.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return text_image()


MENAGERIE_ENTRIES = [("MASTER-ResNetExtra", "build", "example_input", "2019", "OCR")]
