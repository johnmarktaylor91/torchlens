"""Every visual language slide still shows the marks its key claims.

Each slide of ``scripts/visual_language/slides.py`` is drawn with its exact TorchLens call
and every key selector is run over the panel's DOT (Graphviz ``-Tjson``): a renderer change
that silently removes a taught mark fails the slide that teaches it. Legend rows the
coverage map says a slide shows (``coverage.LEGEND_ROW_WITNESS``) must appear in that
slide's DOT text. Semantic assertions only; no golden SVG bytes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.visual_language import capture, coverage
from scripts.visual_language.deck import visible_text
from scripts.visual_language.dot import parse_json, select
from scripts.visual_language.slides import SLIDES, Slide, fill

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.skipif(shutil.which("dot") is None, reason="needs the Graphviz dot binary"),
]

_DRAWN = [slide for slide in SLIDES if slide.panels]


def _render(slide: Slide, out: Path) -> dict[str, Path]:
    """Render every panel of ``slide`` into ``out``; return panel name to JSON path."""

    paths = {}
    for panel in slide.panels:
        capture.render_panel(slide, panel, out)
        paths[panel.name] = out / f"{slide.id}-{panel.name}.json"
    return paths


@pytest.mark.parametrize("slide", _DRAWN, ids=[slide.id for slide in _DRAWN])
def test_slide_keys_find_their_marks(slide: Slide, tmp_path: Path) -> None:
    """Every key and grid cell selector matches at least one mark on its panel."""

    paths = _render(slide, tmp_path)
    selectors = [(k.panel, k.select) for k in slide.keys if k.select]
    selectors += [(c.panel, c.select) for c in slide.cells if c.select]
    missing = [
        f"{panel}: {selector}"
        for panel, selector in selectors
        if not select(parse_json(paths[panel].read_text()), fill(selector))
    ]
    assert not missing, f"slide {slide.id} lost taught marks: {missing}"
    witnessed = [
        text for text, slide_id in coverage.LEGEND_ROW_WITNESS.items() if slide_id == slide.id
    ]
    if witnessed:
        shown = " ".join(visible_text(path) for path in paths.values())
        absent = [text for text in witnessed if text not in shown]
        assert not absent, f"slide {slide.id} does not show legend rows {absent}"
