"""ppdet_centernet exact-name classic.

Paper: Objects as Points.

This module exposes the existing PaddleDetection CenterNet compact
reconstruction under the dependency-gated catalog name.
"""

from __future__ import annotations

from menagerie.classics.paddledet_centernet import build as build
from menagerie.classics.paddledet_centernet import example_input as example_input

__all__ = ["MENAGERIE_ENTRIES", "build", "example_input"]

MENAGERIE_ENTRIES = [("ppdet_centernet", "build", "example_input", "2019", "DET")]
