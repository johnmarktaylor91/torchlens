"""Compatibility module for the AllegroModel target."""

from __future__ import annotations

from torch import nn

from .allegro import build
from .allegro import example_input


MENAGERIE_ENTRIES = [
    ("allegro_AllegroModel", "build", "example_input", "2024", "DE"),
]
