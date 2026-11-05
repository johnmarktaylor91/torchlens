"""Pins for the OKLab colour ramps and the plain encoding key.

The ramp is a stop list blended in OKLab at even perceptual speed, so equal
value steps read as equal colour steps; the drawn key is a labelled colour
bar plus one line per other channel, and the method details ride in the
graph comment, never drawn.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._encoding import interpolate_hex
from torchlens._visualization._oklab import hex_to_oklab, oklab_to_hex, stop_positions
from torchlens._visualization._ramps import (
    DEFAULT_RAMP_FAMILY,
    RAMP_FAMILIES,
    VIRIDIS_9,
    YLORBR_9,
)
from torchlens._visualization.themes import THEME_PRESETS


@pytest.mark.parametrize("color", (*YLORBR_9, *VIRIDIS_9, "#FFFFFF", "#000000", "#C93F3F"))
def test_oklab_round_trips_every_stop(color: str) -> None:
    """hex -> OKLab -> hex is exact on every shipped stop."""

    assert oklab_to_hex(hex_to_oklab(color)) == color


def test_interpolation_hits_stops_and_clamps() -> None:
    """The ends are the end stops; out-of-range fractions clamp."""

    stops = RAMP_FAMILIES["ylorbr"][0]
    assert interpolate_hex(stops, 0.0) == stops[0]
    assert interpolate_hex(stops, 1.0) == stops[-1]
    assert interpolate_hex(stops, -3.0) == stops[0]
    assert interpolate_hex(stops, 7.0) == stops[-1]
    for stop, position in zip(stops, stop_positions(stops), strict=True):
        assert interpolate_hex(stops, position) == stop


@pytest.mark.parametrize("family", sorted(RAMP_FAMILIES))
def test_ramps_step_evenly_and_monotonically(family: str) -> None:
    """Equal fraction steps are near-equal OKLab steps and lightness is monotone.

    On the light page lightness falls (a high value is dark); on the dark
    page it rises (a high value is bright).
    """

    light, dark = RAMP_FAMILIES[family]
    for stops, falling in ((light, True), (dark, False)):
        samples = [hex_to_oklab(interpolate_hex(stops, i / 32)) for i in range(33)]
        steps = [math.dist(a, b) for a, b in zip(samples, samples[1:], strict=False)]
        assert max(steps) < 1.5 * min(steps), (family, falling, steps)
        lightness = [lab[0] for lab in samples]
        ordered = sorted(lightness, reverse=falling)
        assert lightness == ordered, (family, falling)


def test_default_family_feeds_the_skins() -> None:
    """The warm default skins take the default family; viridis is a one-constant flip."""

    assert DEFAULT_RAMP_FAMILY == "ylorbr"
    light, dark = RAMP_FAMILIES[DEFAULT_RAMP_FAMILY]
    for name in ("torchlens", "colorblind"):
        assert THEME_PRESETS[name].ramp == light
    assert THEME_PRESETS["dark"].ramp == dark
    # The light YlOrBr list drops the near-white first class (white-on-white).
    assert light == YLORBR_9[1:]
    # Viridis on the light page: slow (high) is dark purple.
    assert RAMP_FAMILIES["viridis"][0][-1] == "#440154"


class _Tiny(nn.Module):
    """Three ops with different output sizes."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(4, 64)
        self.down = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


@pytest.fixture(scope="module")
def tiny_log() -> Any:
    log = tl.trace(_Tiny(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def test_plain_key_draws_bar_and_size_line(tiny_log: tl.Trace, tmp_path: Path) -> None:
    """color_by + size_by draw a labelled bar and one size line, no method wording."""

    dot = tiny_log.draw(
        color_by="bytes", size_by="bytes", path=str(tmp_path / "g"), file_format="svg"
    ).dot
    graph_line = next(line for line in dot.splitlines() if line.strip().startswith("graph ["))
    comment, drawn = graph_line.split(" label=", 1)
    state = tiny_log._last_encoding_state
    assert drawn.count('FIXEDSIZE="TRUE" WIDTH="5" HEIGHT="10" BGCOLOR=') == 16
    assert f'BGCOLOR="{state.ramp_color(0.0)}"' in drawn
    assert f'BGCOLOR="{state.ramp_color(1.0)}"' in drawn
    assert "output bytes: " in drawn
    assert "size: output bytes, " in drawn and " ... " in drawn
    for jargon in ("min-max", "eligible", "clamped", "size_by:", "color_by:", "sqrt("):
        assert jargon not in drawn, jargon
    # The method record: every former key line, in the graph comment.
    assert "encoding method: color_by: bytes | linear min-max | encoded " in comment
    assert "size ~ sqrt(bytes), min-max, clamped" in comment
    svg = (tmp_path / "g.svg").read_text(encoding="utf-8")
    assert "size ~ sqrt(bytes)" in svg  # the SVG keeps it as an XML comment
    assert "<!-- encoding method" in svg


def test_show_legend_false_draws_no_key_but_keeps_the_record(
    tiny_log: tl.Trace, tmp_path: Path
) -> None:
    """show_legend=False hides the key; the method record stays in the source."""

    dot = tiny_log.draw(
        color_by="bytes", show_legend=False, path=str(tmp_path / "g"), file_format="svg"
    ).dot
    assert 'WIDTH="5" HEIGHT="10" BGCOLOR=' not in dot
    assert "encoding method: color_by: bytes" in dot
