"""F12 pins: live skin palettes (N4), ramps, the N17 neutral fill, and the
CVD accessibility gates.

FORK-2 discipline: the machinery is LIVE but every shipped skin keeps
pixel-identical palettes (the render-identity golden pins them); the
Okabe-Ito set ships as data with its accessibility evidence attached, and
the legacy set's deuteranopia failure is pinned here as the fork's
evidence.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses.audit import palette_distinguishability
from torchlens.visualization.node_spec import NodeSpec
from torchlens.visualization.themes import (
    LEGACY_SEMANTIC_PALETTE,
    OKABE_ITO_SEMANTIC_PALETTE,
    THEME_PRESETS,
    apply_theme_to_spec,
    resolve_theme,
)

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)


def test_every_skin_carries_the_new_records() -> None:
    """Palette, 3-anchor ramp, and neutral fill on all five skins."""

    assert set(THEME_PRESETS) == {"torchlens", "paper", "dark", "colorblind", "high_contrast"}
    role_keys = set(LEGACY_SEMANTIC_PALETTE)
    for theme in THEME_PRESETS.values():
        assert set(theme.semantic_palette) == role_keys, theme.name
        assert len(theme.ramp) == 3
        assert theme.neutral_aggregate_fill.startswith("#")


def test_semantic_palette_default_is_the_shared_read_only_legacy_proxy() -> None:
    """The default palette is the legacy proxy itself, shared and immutable.

    Regression: a plain ``field(default=<mappingproxy>)`` raised ``ValueError:
    mutable default`` at import on Python 3.11, which broke every test module
    that imported the visualization package.
    """

    import dataclasses
    from types import MappingProxyType

    from torchlens.visualization.themes import VisualizationTheme

    theme = VisualizationTheme(
        name="probe",
        graph={},
        node={},
        edge={},
        default_fill="white",
        default_border="black",
        default_font="black",
    )
    assert theme.semantic_palette is LEGACY_SEMANTIC_PALETTE
    assert isinstance(theme.semantic_palette, MappingProxyType)
    with pytest.raises(TypeError):
        theme.semantic_palette["input"] = "#000000"  # type: ignore[index]
    palette_field = {f.name: f for f in dataclasses.fields(VisualizationTheme)}["semantic_palette"]
    assert palette_field.default is dataclasses.MISSING


def test_legend_items_field_is_deleted() -> None:
    """The dead legend_items data is gone (memo skins bullet)."""

    theme = resolve_theme("torchlens")
    assert not hasattr(theme, "legend_items")


def test_low_ramp_anchor_never_equals_default_fill() -> None:
    """The measured white-on-white absence defect cannot recur."""

    for theme in THEME_PRESETS.values():
        assert theme.ramp[0].lower() != str(theme.default_fill).lower(), theme.name


def test_neutral_fill_collides_with_no_semantic_meaning() -> None:
    """N17: the aggregate fill never equals a role colour in the same skin."""

    for theme in THEME_PRESETS.values():
        semantic = {color.lower() for color in theme.semantic_palette.values()}
        assert theme.neutral_aggregate_fill.lower() not in semantic, theme.name


def test_default_skin_palette_is_pixel_identical() -> None:
    """FORK-2 pending: the default skin translates every constant to itself."""

    theme = resolve_theme("torchlens")
    for constant in ("#98FB98", "#ff9999", "#D9D9D9", "#F7D460"):
        spec = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor=constant), theme)
        assert spec.fillcolor == constant


def test_dark_skin_param_remap_is_byte_identical_to_history() -> None:
    """The historical dark param remap now rides the palette, unchanged."""

    dark = resolve_theme("dark")
    flat = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor="#D9D9D9"), dark)
    assert flat.fillcolor == "#374151"
    gradient = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor="#D9D9D9:#B0B0B0"), dark)
    assert gradient.fillcolor == "#374151:#4B5563"
    passthrough = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor="#98FB98"), dark)
    assert passthrough.fillcolor == "#98FB98"


def test_palette_wiring_is_live() -> None:
    """A hypothetical skin palette actually changes the rendered fill: the
    machinery is live, only the shipped palettes are identity."""

    from dataclasses import replace

    flipped = replace(resolve_theme("torchlens"), semantic_palette=OKABE_ITO_SEMANTIC_PALETTE)
    spec = apply_theme_to_spec(NodeSpec(lines=["x"], fillcolor="#98FB98"), flipped)
    assert spec.fillcolor == "#009E73"


def test_okabe_ito_set_passes_all_four_cvd_gates() -> None:
    """The FORK-2 branch-A set is distinguishable under every simulation."""

    findings = palette_distinguishability(dict(OKABE_ITO_SEMANTIC_PALETTE))
    assert len(findings) == 4
    failed = [finding for finding in findings if not finding.passed]
    assert not failed, [finding.detail for finding in failed]


def test_legacy_palette_fails_deuteranopia_the_fork_evidence() -> None:
    """The measured defect: the legacy red/green input/output pair is
    confusable under deuteranopia. Pinned as FORK-2's evidence -- if this
    starts PASSING the fork's premise changed and JMT should hear about it."""

    findings = {
        finding.check: finding
        for finding in palette_distinguishability(dict(LEGACY_SEMANTIC_PALETTE))
    }
    assert not findings["cvd_deuteranopia"].passed
    assert findings["cvd_deuteranopia"].measurements["collisions"]
    # The near-zero grayscale collision (measured boolean~input 1.5) also fails.
    assert not findings["cvd_grayscale"].passed


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
            vis_outpath=str(tmp_path / "neutral"),
            vis_fileformat="svg",
            vis_save_only=True,
            return_graph=True,
        )
        node_lines = [
            line
            for line in graph.source.splitlines()
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
        assert resolution.draw_kwargs["vis_theme"] == skin
    finally:
        log.cleanup()
