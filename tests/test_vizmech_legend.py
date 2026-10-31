"""The key sits under the header, top left, and draws real shapes (JMT, 29 Oct 2026).

Pins here:

- the forward key is part of the graph label, header first, then the key,
  so no layout engine can float it to another corner and a top crop keeps
  both; there is no separate legend node;
- each key entry draws the mark's real shape (oval, box, cylinder) and the
  fill the picture paints, including the dark skin's remapped weight grays;
- entries appear only for marks the picture draws; both weight grays appear
  whenever weights are drawn; plain operations and arrows get no entry;
- the key is off by default (``show_legend=None``) and the header stays;
- the backward-vocabulary key builder emits a row per ACTIVE style only,
  per the WGAN-GP acceptance contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._header_block import KEY_TABLE_ID, key_swatches
from torchlens._visualization._legend import LEGEND_NODE_NAME, backward_key_sections
from torchlens._visualization.themes import THEME_PRESETS


class _Nested(nn.Module):
    """A weighted leaf module inside a nested module, then a plain add."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


def _draw_dot(tmp_path: Path, name: str, **kwargs: object) -> str:
    """Render a tiny model to SVG and return the DOT source."""

    trace = tl.trace(_Nested(), torch.randn(1, 3))
    return trace.draw(
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / name),
        **kwargs,
    )


def _graph_label(dot: str) -> str:
    """Return the graph label statement of a DOT source."""

    return next(
        line
        for line in dot.splitlines()
        if line.strip().startswith("graph [") and "label=<" in line
    )


def test_key_sits_in_the_header_block(tmp_path: Path) -> None:
    """Header first, then the key, inside the one top-left graph label."""

    dot = _draw_dot(tmp_path, "key_on", show_legend=True)
    assert LEGEND_NODE_NAME not in dot  # no free-floating legend node
    label = _graph_label(dot)
    assert "labelloc=t" in dot and "labeljust=left" in dot
    assert label.index("_Nested") < label.index(f'ID="{KEY_TABLE_ID}"')
    for word in ("input", "output", "trainable weights", "frozen weights", "module"):
        assert f">{word}<" in label, f"key entry {word!r} missing"
    for absent in ("plain op", "operation", "arrow", "buffer", "true/false test"):
        assert f">{absent}<" not in label
    svg = (tmp_path / "key_on.svg").read_text()
    assert ">trainable weights<" in svg
    assert "<image" in svg  # the swatches are drawn shapes, not bare colour cells


def test_key_swatches_are_real_shapes_and_files_exist(tmp_path: Path) -> None:
    """Input and output swatches are ovals, weights are boxes, a module is a module frame."""

    dot = _draw_dot(tmp_path, "key_shapes", show_legend=True)
    sources = re.findall(r'IMG SRC="([^"]+)"', _graph_label(dot))
    assert sources and all(Path(src).is_absolute() and Path(src).is_file() for src in sources)
    shapes = [Path(src).name.split("-")[1] for src in sources]
    assert shapes == ["oval", "oval", "box", "box", "frame"]


def test_no_key_by_default(tmp_path: Path) -> None:
    """AUTO (None) with no channel draws the header and no key."""

    dot = _draw_dot(tmp_path, "key_auto")
    assert KEY_TABLE_ID not in dot
    assert LEGEND_NODE_NAME not in dot
    assert "_Nested" in dot  # the header stays


def test_dark_swatches_show_rendered_weight_grays() -> None:
    """The dark skin's remapped weight grays are what the key shows."""

    dark = THEME_PRESETS["dark"]
    swatches = {
        s.text: s for s in key_swatches(dark, frozenset({"parameterized"}), has_module=False)
    }
    assert swatches["trainable weights"].fill == "#374151"
    assert swatches["frozen weights"].fill == "#6B7280"
    assert "#56B4E9" not in {s.fill for s in swatches.values()}


def test_key_entries_follow_drawn_marks() -> None:
    """Buffer, test, mutated parameter and intervention entries need their mark."""

    theme = THEME_PRESETS["torchlens"]
    bare = key_swatches(theme, frozenset({"input", "output"}), has_module=False)
    assert [s.text for s in bare] == ["input", "output"]
    full = key_swatches(
        theme,
        frozenset(
            {
                "input",
                "output",
                "parameterized",
                "buffer",
                "boolean",
                "edited",
                "changed by the edit",
            }
        ),
        has_module=True,
        mutated_parameter=True,
    )
    assert [s.shape for s in full if s.text in {"buffer", "mutated parameter"}] == [
        "cylinder",
        "cylinder",
    ]
    assert [s.text for s in full][-3:] == ["true/false test", "edited", "changed by the edit"]
    edited, changed = full[-2], full[-1]
    assert (edited.shape, edited.border, edited.line) == ("oval", "#7B1FA2", 2.5)
    assert changed.shape == "arrow"
    assert changed.fill == "#7B1FA240"
    module = next(s for s in full if s.text == "module")
    assert module.shape == "frame"
    assert module.line > 1.0


def test_backward_key_rows_gate_on_active_styles() -> None:
    """A key row for an unused style would itself be an unexplained claim."""

    minimal = backward_key_sections(
        has_higher_order=False,
        has_intervening=False,
        has_accumulation=False,
        has_custom=False,
        num_backward_passes=1,
    )
    texts = [row.text for section in minimal for row in section.rows]
    assert any("backward op" in text for text in texts)
    assert not any("grad-of-grad" in text for text in texts)
    assert not any("accum" in text for text in texts)
    full = backward_key_sections(
        has_higher_order=True,
        has_intervening=True,
        has_accumulation=True,
        has_custom=True,
        num_backward_passes=2,
    )
    full_texts = [row.text for section in full for row in section.rows]
    assert any("grad-of-grad" in text for text in full_texts)
    assert any("[i]" in text for text in full_texts)
    assert any("accum" in text for text in full_texts)
    assert any("bwd N" in text for text in full_texts)


def test_rank_path_key_rides_the_header(tmp_path: Path) -> None:
    """The rank engine draws the same header block, no pinned legend node."""

    trace = tl.trace(nn.Sequential(nn.Linear(2, 2), nn.ReLU()), torch.ones(1, 2))
    dot = trace.draw(
        path=str(tmp_path / "rank_key"),
        file_format="svg",
        vis_save_only=True,
        vis_node_placement="rank",
        show_legend=True,
    )
    assert LEGEND_NODE_NAME not in dot
    assert f'ID="{KEY_TABLE_ID}"' in dot
    assert ">trainable weights<" in dot


def test_output_has_no_double_outline(tmp_path: Path) -> None:
    """The double outline stays reserved for hidden buffer parents, never the output."""

    dot = _draw_dot(tmp_path, "output_outline")
    output_lines = [
        line
        for line in dot.splitlines()
        if line.strip().startswith("output_1") and "[" in line  # node statements, not rank lists
    ]
    assert output_lines
    assert all("peripheries" not in line for line in output_lines)
    assert all("shape=oval" in line for line in output_lines)  # a function call stays an oval
