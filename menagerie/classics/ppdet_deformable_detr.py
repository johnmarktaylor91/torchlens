"""ppdet_deformable_detr exact-name classic.

Paper: Deformable DETR: Deformable Transformers for End-to-End Object Detection.

The imported compact model keeps multi-scale deformable attention and object
queries from the PaddleDetection Deformable DETR reconstruction.
"""

from __future__ import annotations

from menagerie.classics.paddledet_deformable_detr import build as build
from menagerie.classics.paddledet_deformable_detr import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_deformable_detr", "build", "example_input", "2020", "DET")]
