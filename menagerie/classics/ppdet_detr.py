"""ppdet_detr exact-name classic.

Paper: End-to-End Object Detection with Transformers.

The imported compact model keeps the CNN feature map, learned object queries,
transformer decoder, and set-prediction heads from DETR.
"""

from __future__ import annotations

from menagerie.classics.paddledet_detr import build as build
from menagerie.classics.paddledet_detr import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_detr", "build", "example_input", "2020", "DET")]
