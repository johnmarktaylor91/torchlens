"""Compatibility module for SRFormer-light x4.

Paper: "SRFormer: Permuted Self-Attention for Single Image Super-Resolution",
Zhou et al., ICCV 2023.

The implementation lives in :mod:`menagerie.classics.srformer_x4` so the full
and light variants share the same compact PSA reconstruction.
"""

from __future__ import annotations

from menagerie.classics.srformer_x4 import build_srformer_light_x4, example_input

MENAGERIE_ENTRIES = [
    (
        "SRFormer-light x4 (permuted self-attention SR)",
        "build_srformer_light_x4",
        "example_input",
        "2023",
        "E7",
    )
]

__all__ = ["MENAGERIE_ENTRIES", "build_srformer_light_x4", "example_input"]
