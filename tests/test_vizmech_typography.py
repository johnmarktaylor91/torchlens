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

import pytest
import torch
from torch import nn

from torchlens._visualization._typography import (
    DEFAULT_TYPOGRAPHY,
    TypographyRecord,
)
from torchlens._visualization.themes import (
    THEME_PRESETS,
    theme_edge_attrs,
    theme_graph_attrs,
    theme_node_attrs,
)


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


def test_scaled_preserves_ratios() -> None:
    """``scaled()`` is the one door for size scaling; ratios are invariant."""

    scaled = DEFAULT_TYPOGRAPHY.scaled(1.5)
    assert scaled.base_size == pytest.approx(21.0)
    assert scaled.annotation_ratio == pytest.approx(DEFAULT_TYPOGRAPHY.annotation_ratio)
    assert scaled.secondary_ratio == pytest.approx(DEFAULT_TYPOGRAPHY.secondary_ratio)
    assert scaled.emphasis_ratio == pytest.approx(DEFAULT_TYPOGRAPHY.emphasis_ratio)


class _Tiny(nn.Module):
    """Two-op model, cheap enough for a smoke-tier rank render."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))
