"""The deck's "At scale and edge cases" section draws real architectures.

The rows are fixed (``scripts/visual_language/at_scale.py``); a row whose model the deck cannot
build offline is pending with its reason. The render test draws every panel of every active
row through Graphviz on a torchvision model with random weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.visual_language.at_scale import SCALE_ROWS, scale_slides


def test_rows_are_fixed_and_pending_rows_say_why() -> None:
    """The section's ids are stable, real architectures only, and pending rows name the gap."""

    assert [row.id for row in SCALE_ROWS] == [
        "scale-resnet18",
        "scale-resnet50",
        "scale-densenet121",
        "scale-gpt2",
        "scale-generation",
    ]
    for row in SCALE_ROWS:
        assert (row.status == "active") == bool(row.panels), row.id
        assert row.status == "active" or row.reason, row.id
    assert [s.id for s in scale_slides()] == [r.id for r in SCALE_ROWS if r.status == "active"]


def _cells() -> list:
    out = []
    for slide in scale_slides():
        # DenseNet-121 unrolled is the largest graph in the deck: its cell is slow-tier.
        marks = [pytest.mark.slow] if "densenet" in slide.id else []
        out.append(pytest.param(slide, id=slide.id, marks=marks))
    return out


@pytest.mark.heavy
@pytest.mark.parametrize("slide", _cells())
def test_every_active_row_renders(slide, tmp_path: Path) -> None:
    """Every panel draws a non-empty DOT graph and an SVG picture."""

    pytest.importorskip("torchvision", reason="the at-scale models are torchvision's")
    from scripts.visual_language import capture

    for panel in slide.panels:
        record = capture.render_panel(slide, panel, tmp_path)
        stem = record["stem"]
        assert "digraph" in (tmp_path / f"{stem}.dot").read_text(), stem
        assert (tmp_path / f"{stem}.svg").stat().st_size > 0, stem
