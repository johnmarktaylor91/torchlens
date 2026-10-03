"""Vizmech wave-2 item 13: one compact legend table in a dedicated rank (D28).

The historical legend was six disconnected nodes dot packed BESIDE the model
(366 -> 1048 pt page width, 2.86x, zero collisions -- the two-oracles worked
example). Pins here:

- the role legend renders as ONE plaintext HTML-table node, placed in a
  ``rank=sink`` group and tied into the model's component with an invisible
  non-constraint edge (no more disconnected-component packing);
- swatches carry the RENDERED fills (dark theme remaps), never the
  aspirational palette hexes;
- the channel (AUTO) disclosure and the role legend are SECTIONS of the one
  table -- a render never carries two legend topologies;
- the backward-vocabulary key builder (item 17 groundwork) emits a row per
  ACTIVE style only, per the WGAN-GP acceptance contract.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._legend import (
    LEGEND_NODE_NAME,
    backward_key_sections,
    build_legend_table_label,
    theme_role_sections,
)
from torchlens.visualization.themes import THEME_PRESETS


def _draw_dot(tmp_path: Path, name: str, **kwargs: object) -> str:
    """Render a tiny model to SVG and return the DOT source."""

    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(1, 3))
    return trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / name),
        **kwargs,
    )


def test_role_legend_is_one_table_node_in_dedicated_rank(tmp_path: Path) -> None:
    """One plaintext node, rank=sink, invis tie -- never six free nodes."""

    dot = _draw_dot(tmp_path, "legend_on", show_legend=True)
    assert dot.count("TorchLens legend") == 1
    assert "tl_legend_0" not in dot  # the six-node form is retired
    assert "rank=sink" in dot
    assert f"-> {LEGEND_NODE_NAME} [" in dot  # the component tie
    assert "style=invis" in dot and "constraint=false" in dot
    svg = (tmp_path / "legend_on.svg").read_text()
    assert "TorchLens legend" in svg
    for role in ("input", "output", "parameterized", "boolean"):
        assert f">{role}<" in svg, f"role row {role!r} missing from the rendered table"
    # The buffer row carries its shape-vocabulary hint inline.
    assert ">buffer (cylinder)<" in svg


def test_no_legend_by_default(tmp_path: Path) -> None:
    """AUTO (None) with no channel emits nothing."""

    dot = _draw_dot(tmp_path, "legend_auto")
    assert LEGEND_NODE_NAME not in dot


def test_swatches_show_rendered_colors_not_palette_hexes() -> None:
    """The dark theme's remapped parameterized fill is what the legend shows."""

    dark = THEME_PRESETS["dark"]
    label = build_legend_table_label(theme_role_sections(dark), dark)
    assert "#374151" in label  # the dark-remapped parameterized fill
    assert "#56B4E9" not in label  # the aspirational palette hex


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


def test_rank_path_legend_is_one_pinned_table(tmp_path: Path) -> None:
    """The rank path emits the same one-table form, pinned for neato -n."""

    trace = tl.trace(nn.Sequential(nn.Linear(2, 2), nn.ReLU()), torch.ones(1, 2))
    dot = trace.draw(
        vis_outpath=str(tmp_path / "rank_legend"),
        vis_fileformat="svg",
        vis_save_only=True,
        vis_node_placement="rank",
        show_legend=True,
    )
    assert dot.count("TorchLens legend") == 1
    assert "tl_legend_0" not in dot
    assert f"{LEGEND_NODE_NAME} [" in dot
    # neato -n requires every node positioned: the table is pinned.
    legend_line = next(line for line in dot.splitlines() if line.strip().startswith("tl_legend"))
    assert 'pos="' in legend_line and legend_line.rstrip().endswith('!"]')
