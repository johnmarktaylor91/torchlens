"""F12 pins: live skin palettes (N4), ramps, the N17 neutral fill, and the
CVD accessibility gates.

The default ``torchlens`` skin and ``colorblind`` render the Okabe-Ito set
(the flip was held until the collapse budget gate landed); ``paper`` and
``high_contrast`` keep the legacy role colours, and the legacy set's
deuteranopia failure is pinned here as the fork's evidence.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import lenses
from torchlens._visualization.lenses.audit import palette_distinguishability
from torchlens._visualization.node_spec import NodeSpec
from torchlens._visualization.themes import (
    LEGACY_SEMANTIC_PALETTE,
    OKABE_ITO_SEMANTIC_PALETTE,
    THEME_PRESETS,
    apply_theme_to_spec,
    resolve_theme,
)


# Kept: the slices removed this test; the stack's Okabe-Ito default-skin work rewrote it to assert the new palette.
def test_default_skin_renders_the_okabe_ito_palette() -> None:
    """The default skin translates every legacy constant to its Okabe-Ito role colour."""

    theme = resolve_theme("torchlens")
    assert theme.semantic_palette is OKABE_ITO_SEMANTIC_PALETTE
    for role, constant in LEGACY_SEMANTIC_PALETTE.items():
        spec = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor=constant), theme)
        assert spec.fillcolor == OKABE_ITO_SEMANTIC_PALETTE[role], role
    assert resolve_theme("colorblind").semantic_palette is OKABE_ITO_SEMANTIC_PALETTE


# Kept: the slices removed this test; the stack's Okabe-Ito default-skin work rewrote it to assert the new palette.
def test_paper_and_high_contrast_keep_their_historical_palettes() -> None:
    """The paper and high_contrast skins keep the legacy role colours."""

    for name in ("paper", "high_contrast"):
        theme = resolve_theme(name)
        assert theme.semantic_palette is LEGACY_SEMANTIC_PALETTE
        spec = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor="#98FB98"), theme)
        assert spec.fillcolor == "#98FB98", name


def test_legacy_palette_fails_deuteranopia_the_fork_evidence() -> None:
    """The measured defect: the legacy red/green input/output pair is
    confusable under deuteranopia. Pinned as FORK-2's evidence -- if this
    starts PASSING the fork's premise changed and the maintainer should hear about it."""

    findings = {
        finding.check: finding
        for finding in palette_distinguishability(dict(LEGACY_SEMANTIC_PALETTE))
    }
    assert not findings["cvd_deuteranopia"].passed
    assert findings["cvd_deuteranopia"].measurements["collisions"]
    # The near-zero grayscale collision (measured boolean~input 1.5) also fails.
    assert not findings["cvd_grayscale"].passed


@pytest.mark.smoke
def test_neutral_collapsed_fill_under_active_channel(tmp_path: Any) -> None:
    """N17 end-to-end: channel + compaction paints collapsed boxes neutral
    and discloses it (composition row 13)."""

    class Blocks(nn.Module):
        """Two boxed blocks so collapse has something to box."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
            self.b = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))

        def forward(self, x: Any) -> Any:
            return self.b(self.a(x))

    log = tl.trace(Blocks(), torch.randn(2, 8))
    try:
        resolution = lenses.resolve_lens(log, "speed", {"collapse": 1.0})
        assert "collapsed_node_spec_fn" in resolution.draw_kwargs
        assert any("aggregate, not encoded" in line for line in resolution.disclosure)
        # The composed collapsed fn paints the declared neutral appearance.
        theme = resolve_theme("torchlens")
        spec = NodeSpec(lines=["box"], fillcolor="#D9D9D9")
        painted = resolution.draw_kwargs["collapsed_node_spec_fn"](None, spec)
        assert painted.fillcolor == theme.neutral_aggregate_fill
        assert "aggregate, not encoded" in painted.lines
        # Exclusivity (composition row 13): with the channel active, no
        # rendered node keeps the trainable-params grey that means something
        # else in the same picture.
        graph = log.draw(
            **resolution.draw_kwargs,
            path=str(tmp_path / "neutral"),
            file_format="svg",
        ).dot
        node_lines = [
            line
            for line in graph.splitlines()
            if "tl_legend_" not in line and "tl_encoding_legend_" not in line
        ]
        assert not any('fillcolor="#D9D9D9"' in line for line in node_lines)
    finally:
        log.cleanup()


@pytest.mark.parametrize("skin", sorted(THEME_PRESETS))
def test_skins_compose_with_lenses(skin: str, tmp_path: Any) -> None:
    """Every skin resolves under a lens (the two axes compose freely)."""

    log = tl.trace(nn.Linear(4, 4), torch.randn(1, 4))
    try:
        resolution = lenses.resolve_lens(log, "overview", skin=skin)
        assert resolution.draw_kwargs["theme"] == skin
    finally:
        log.cleanup()
