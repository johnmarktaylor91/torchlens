"""Compatibility module for the Allegro OEQ target."""

from __future__ import annotations

from torch import nn

from .allegro import build_small as build
from .allegro import example_input


MENAGERIE_ENTRIES = [
    ("allegro_Allegro_oeq", "build", "example_input", "2024", "DE"),
]
