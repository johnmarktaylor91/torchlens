"""ppdet_yolof exact-name classic.

Paper: You Only Look One-level Feature.

The imported compact model keeps the dilated encoder and one-level dense
prediction head.
"""

from __future__ import annotations

from menagerie.classics.paddledet_yolof import build_paddledet_yolof as build
from menagerie.classics.paddledet_yolof import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_yolof", "build", "example_input", "2021", "DET")]
