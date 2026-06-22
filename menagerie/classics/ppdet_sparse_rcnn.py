"""ppdet_sparse_rcnn exact-name classic.

Paper: Sparse R-CNN: End-to-End Object Detection with Learnable Proposals.

The imported compact model keeps learned proposal boxes/features and iterative
dynamic proposal heads.
"""

from __future__ import annotations

from menagerie.classics.paddledet_sparse_rcnn import build_paddledet_sparse_rcnn as build
from menagerie.classics.paddledet_sparse_rcnn import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_sparse_rcnn", "build", "example_input", "2021", "DET")]
