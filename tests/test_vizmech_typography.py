"""Vizmech wave-2 item 14: the one typography record (M(vizmech) D29).

Pins:
- every theme preset carries a pinned font family on ALL THREE Graphviz
  scopes (graph covers cluster captions and the graph caption) -- before the
  fix the default preset pinned nothing and was the only serif theme, and a
  theme swap alone moved measured label geometry (59 -> 68 violations);
- the attribute builders inject the family into any scope a preset leaves
  unset (defense against a future preset regressing to the serif fallback);
- the historical literal size channels (8 / 10 / 18 pt) consume the record --
  a source lint keeps new literals from creeping back;
- the rank path emits the same family on graph/node/edge scopes (it emitted
  NONE of them before, so rank renders drew in Times whatever the theme said).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.visualization._rank_layout_internal.layout as layout_mod
from torchlens.visualization._edge_multiplicity import _MULTIPLICITY_LABEL_FONT_SIZE
from torchlens.visualization._render_common import _EDGE_LABEL_FONT_SIZE
from torchlens.visualization._typography import (
    DEFAULT_TYPOGRAPHY,
    TypographyRecord,
    format_pt,
)
from torchlens.visualization.themes import (
    THEME_PRESETS,
    theme_edge_attrs,
    theme_graph_attrs,
    theme_node_attrs,
)

_VIZ_PACKAGE = Path(tl.visualization.__file__).parent


def test_every_preset_pins_fontname_on_all_scopes() -> None:
    """No preset may leave any scope on the engine's serif fallback (D29)."""

    for name, theme in THEME_PRESETS.items():
        for scope_name, scope in (
            ("graph", theme.graph),
            ("node", theme.node),
            ("edge", theme.edge),
        ):
            assert "fontname" in scope, (
                f"theme {name!r} leaves the {scope_name} scope without a pinned "
                "fontname -- the engine falls back to a serif family and label "
                "geometry moves by theme swap alone"
            )


def test_attr_builders_inject_family_when_scope_unset() -> None:
    """The builders backstop any preset that forgets a scope."""

    bare = TypographyRecord(family="TestFamily")
    theme = THEME_PRESETS["torchlens"].__class__(
        name="bare",
        graph={},
        node={},
        edge={},
        default_fill="white",
        default_border="black",
        default_font="black",
        typography=bare,
    )
    assert theme_graph_attrs(theme)["fontname"] == "TestFamily"
    assert theme_node_attrs(theme)["fontname"] == "TestFamily"
    assert theme_edge_attrs(theme)["fontname"] == "TestFamily"


def test_explicit_scope_fontname_wins_over_record() -> None:
    """A preset's explicit scope fontname is never overridden."""

    theme = THEME_PRESETS["high_contrast"]
    assert theme_node_attrs(theme)["fontname"] == "Helvetica-Bold"


def test_record_roles_match_historical_literals() -> None:
    """The semantic roles carry the exact historical sizes, formatted stably."""

    assert DEFAULT_TYPOGRAPHY.annotation_pt == "8"
    assert DEFAULT_TYPOGRAPHY.secondary_pt == "10"
    assert DEFAULT_TYPOGRAPHY.emphasis_pt == "18"
    assert _EDGE_LABEL_FONT_SIZE == 8
    assert _MULTIPLICITY_LABEL_FONT_SIZE == 8
    assert format_pt(8.5) == "8.5"


def test_scaled_preserves_ratios() -> None:
    """``scaled()`` is the one door for size scaling; ratios are invariant."""

    scaled = DEFAULT_TYPOGRAPHY.scaled(1.5)
    assert scaled.base_size == pytest.approx(21.0)
    assert scaled.annotation_ratio == pytest.approx(DEFAULT_TYPOGRAPHY.annotation_ratio)
    assert scaled.secondary_ratio == pytest.approx(DEFAULT_TYPOGRAPHY.secondary_ratio)
    assert scaled.emphasis_ratio == pytest.approx(DEFAULT_TYPOGRAPHY.emphasis_ratio)


def test_no_literal_font_size_channels_remain() -> None:
    """Source lint: the scattered literal channels stay retired.

    The record is the only place the 8/10/18 sizes may be spelled; a new
    ``labelfontsize="8"``-style literal in the visualization package is the
    defect class this item removed (vizmech defect 15).
    """

    literal_patterns = (
        re.compile(r"labelfontsize[\"']?\s*[:=]\s*[\"']\d+[\"']"),
        re.compile(r"POINT-SIZE=\\?[\"']\d+"),
    )
    offenders: list[str] = []
    for source_path in _VIZ_PACKAGE.rglob("*.py"):
        if source_path.name == "_typography.py":
            continue
        text = source_path.read_text()
        for pattern in literal_patterns:
            for match in pattern.finditer(text):
                offenders.append(f"{source_path.name}: {match.group(0)}")
    assert not offenders, (
        "literal font-size channels found; route them through the typography "
        f"record (_typography.py): {offenders}"
    )


class _Tiny(nn.Module):
    """Two-op model, cheap enough for a smoke-tier rank render."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def test_rank_path_emits_theme_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The rank-path DOT carries the pinned family on graph/node/edge scopes."""

    captured: dict[str, str] = {}

    def _fake_neato(**kwargs: object) -> SimpleNamespace:
        captured["source"] = Path(str(kwargs["source_path"])).read_text()
        Path(str(kwargs["rendered_path"])).write_text("<svg></svg>")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(layout_mod, "_run_neato_with_fallbacks", _fake_neato)
    trace = tl.trace(_Tiny(), torch.randn(1, 4))
    trace.draw(
        vis_node_placement="rank",
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "rank_typography"),
        show_containers=False,
    )
    source = captured["source"]
    graph_line = next(line for line in source.splitlines() if line.strip().startswith("graph ["))
    node_line = next(line for line in source.splitlines() if line.strip().startswith("node ["))
    edge_line = next(line for line in source.splitlines() if line.strip().startswith("edge ["))
    for scope_name, line in (("graph", graph_line), ("node", node_line), ("edge", edge_line)):
        assert "fontname=" in line and "Helvetica" in line, (
            f"rank-path {scope_name} scope lost the pinned family: {line.strip()}"
        )
