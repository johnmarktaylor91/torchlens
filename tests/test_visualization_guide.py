"""The visualization guide and its figure script agree.

``docs/guides/visualization.md`` shows one image per row of the figure table in
``docs/guides/visualization_figures.py`` with the exact ``draw`` call that made it.
These checks keep the page, the table, the committed images and ``Trace.draw``'s
signature in lockstep without rendering anything.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType

import torchlens as tl

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "guides" / "visualization.md"
IMAGES = REPO / "docs" / "images" / "visualization"


def _load_figures() -> ModuleType:
    path = REPO / "docs" / "guides" / "visualization_figures.py"
    spec = importlib.util.spec_from_file_location("_viz_guide_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


figures = _load_figures()
PAGE_TEXT = PAGE.read_text(encoding="utf-8")


def test_every_figure_is_on_the_page_with_its_call() -> None:
    for figure in figures.FIGURES:
        image_ref = f"(../images/visualization/{figure.filename})"
        assert image_ref in PAGE_TEXT, f"{figure.filename} is not shown on the page"
        assert f"`{figures.draw_call(figure)}`" in PAGE_TEXT, (
            f"the page does not show {figures.draw_call(figure)} for {figure.name}"
        )


def test_every_page_image_has_a_figure_and_a_file() -> None:
    names = {figure.filename for figure in figures.FIGURES}
    referenced = set(re.findall(r"\.\./images/visualization/([\w.-]+)", PAGE_TEXT))
    assert referenced == names
    on_disk = {path.name for path in IMAGES.iterdir() if path.is_file()}
    assert on_disk == names, "committed images and the figure table differ"


def test_figure_keywords_are_draw_parameters() -> None:
    parameters = inspect.signature(tl.Trace.draw).parameters
    for figure in figures.FIGURES:
        unknown = set(figure.kwargs) - set(parameters)
        assert not unknown, f"{figure.name} uses keywords Trace.draw lacks: {sorted(unknown)}"


def test_figure_names_are_unique() -> None:
    names = [figure.name for figure in figures.FIGURES]
    assert len(names) == len(set(names))
