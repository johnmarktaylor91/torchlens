"""GenCast graph-transformer diffusion weather denoiser alias module.

Paper: Price et al. 2023/2024, "GenCast: Diffusion-based ensemble forecasting
for medium-range weather." This module exposes the existing compact
GraphCast-derived graph-transformer denoiser under the requested GenCast names.
"""

from __future__ import annotations

from menagerie.classics.graphcast_gencast_graphtransformer import build, example_input


MENAGERIE_ENTRIES = [
    ("GenCast", "build", "example_input", "2023", "E7"),
    ("gencast.GenCast", "build", "example_input", "2023", "E7"),
]
