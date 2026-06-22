"""ppdet_dino exact-name classic.

Paper: DINO: DETR with Improved DeNoising Anchor Boxes.

The imported compact model keeps denoising queries, anchor-style query
initialization, and transformer detection heads.
"""

from __future__ import annotations

from menagerie.classics.paddledet_dino import build as build
from menagerie.classics.paddledet_dino import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_dino", "build", "example_input", "2022", "DET")]
