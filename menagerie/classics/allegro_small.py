"""Compatibility module for the compact Allegro small classic."""

from __future__ import annotations

from torch import Tensor, nn

from .allegro import build_small as build
from .allegro import example_input


def build_allegro_small() -> nn.Module:
    """Build the compact Allegro small variant.

    Returns
    -------
    nn.Module
        Random-initialized Allegro small model.
    """
    return build()


MENAGERIE_ENTRIES = [
    ("Allegro small", "build_allegro_small", "example_input", "2024", "DE"),
]
