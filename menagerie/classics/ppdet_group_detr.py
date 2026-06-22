"""ppdet_group_detr exact-name classic.

Paper: Group DETR.

The imported compact model keeps grouped object-query training/inference heads
from the existing PaddleDetection Group DETR reconstruction.
"""

from __future__ import annotations

from menagerie.classics.paddledet_group_detr import build as build
from menagerie.classics.paddledet_group_detr import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_group_detr", "build", "example_input", "2022", "DET")]
