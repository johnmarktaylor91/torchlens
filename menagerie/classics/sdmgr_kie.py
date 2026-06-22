"""SDMGR-KIE: Spatial Dual-Modality Graph Reasoning for documents.

Paper: Sun et al. 2021, "Spatial Dual-Modality Graph Reasoning for Key
Information Extraction."  The compact model keeps text and visual node
features, spatial edge features, graph message passing, and node classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._mmocr_shared import SDMGRKIE, kie_input


def build() -> nn.Module:
    """Build compact SDMGR-KIE.

    Returns
    -------
    nn.Module
        Random-initialized graph reasoner.
    """

    return SDMGRKIE()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact document graph inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Text ids, visual features, and bounding boxes.
    """

    return kie_input()


MENAGERIE_ENTRIES = [("SDMGR-KIE", "build", "example_input", "2021", "KIE")]
