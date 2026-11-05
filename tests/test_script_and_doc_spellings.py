"""Maintainer scripts and backend docs spell the current ``Trace.draw`` and ``save=`` names.

The two render scripts are not run by the test tiers, so a removed draw keyword
(``vis_renderer``, ``vis_mode``, ``vis_call_depth``, ``vis_save_only``) only failed
when a maintainer next regenerated the gallery; this pins their keywords to the
live ``Trace.draw`` signature, and the tinygrad guide to the ``save=`` spelling.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import torchlens as tl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DRAW_SCRIPTS = ("scripts/render_encoding_reference.py",)


def _draw_keywords(source: str) -> dict[str, set[str]]:
    """Return keyword names per ``.draw(`` / ``_render(`` call site, keyed by line."""

    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in {"draw", "_render"}:
            continue
        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        if keywords:
            found[f"line {node.lineno}"] = keywords
    return found


@pytest.mark.parametrize("rel", _DRAW_SCRIPTS)
def test_render_scripts_use_live_draw_keywords(rel: str) -> None:
    """Every keyword a script passes to ``draw`` (directly or via ``_render``) is a live one."""

    live = set(inspect.signature(tl.Trace.draw).parameters) - {"self"}
    source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
    sites = _draw_keywords(source)
    assert sites, f"{rel}: no draw call found"
    stale = {site: sorted(kws - live) for site, kws in sites.items() if kws - live}
    assert not stale, f"{rel} passes removed draw keywords: {stale}"


def test_tinygrad_guide_spells_the_save_option() -> None:
    """The tinygrad implementation guide names the restricted option by its live spelling."""

    guide = (_REPO_ROOT / "torchlens/backends/tinygrad/AGENTS.md").read_text(encoding="utf-8")
    assert "layers_to_save" not in guide
    assert '`save` must be `"all"`' in guide
